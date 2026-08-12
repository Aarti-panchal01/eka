"""One-command driver: trained LoRA adapter -> merged model -> served Ollama
endpoint, optionally tunnelled to the internet with ngrok.

    python serving/merge_lora_and_serve.py --mode founder
    python serving/merge_lora_and_serve.py --all --ngrok
    python serving/merge_lora_and_serve.py --mode founder --skip-merge --gguf
    python serving/merge_lora_and_serve.py --all --dry-run

This is PHASE 7 (serving), not training. It assumes ml/scripts/train_<mode>_lora
_kaggle.py has already run and pushed an adapter to
`<HF_USERNAME>/eka-<mode>-lora` on the Hub. This script does NOT reimplement
merging — that logic already lives in ml/scripts/merge_lora.py and is invoked
here as a subprocess so there is exactly one place that knows how to merge.

Pipeline per mode:
    1. verify prerequisites (HF_TOKEN/HF_USERNAME, adapter exists on the Hub,
       `ollama` on PATH, enough free disk)
    2. merge:      python ml/scripts/merge_lora.py --mode <mode>   (skippable)
    3. [optional]  convert the merged safetensors to GGUF via llama.cpp
    4. register:   ollama create eka-<mode> -f Modelfile
    5. verify:     ollama run eka-<mode> "hello"  (must return non-empty text)
    6. [optional]  start `ngrok http 11434` and print the public URL to paste
                    into Render's OLLAMA_BASE_URL env var

# DEPLOY_LATER: this whole file is post-training serving glue. Run it once a
# mode's adapter exists on the Hub, then flip LLM_MODE=ollama in the backend's
# .env (local) or Render dashboard (remote, via --ngrok's printed URL).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

# Windows consoles default to cp1252, which cannot encode checkmarks/arrows.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SERVING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVING_DIR.parent
ML_DIR = PROJECT_ROOT / "ml"
MERGED_DIR = ML_DIR / "models" / "merged"
MERGE_SCRIPT = ML_DIR / "scripts" / "merge_lora.py"

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

MODES = ("founder", "chanakya", "gita", "reflection")
GB_PER_MODE = 16
NGROK_API = "http://127.0.0.1:4040/api/tunnels"


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def step(text: str) -> None:
    print(f"\n--> {text}")


def ok(text: str) -> None:
    print(f"    OK  {text}")


def warn(text: str) -> None:
    print(f"    !!  {text}")


def fail(text: str) -> None:
    print(f"    XX  {text}")


# ------------------------------------------------------------- prerequisites
def check_env_prereqs() -> Optional[Dict[str, str]]:
    """HF_TOKEN + HF_USERNAME must be set. Returns them, or None + prints why."""
    token = os.environ.get("HF_TOKEN", "").strip()
    username = os.environ.get("HF_USERNAME", "").strip()
    if not token or not username:
        fail(
            "HF_TOKEN and/or HF_USERNAME are not set.\n"
            "        Set them in the project .env (or your shell env) — the same "
            "credentials\n"
            "        training/train_<mode>_lora_kaggle.py used to push the adapter."
        )
        return None
    return {"token": token, "username": username}


def check_adapter_on_hub(mode: str, username: str, token: str) -> bool:
    """Confirm `<username>/eka-<mode>-lora` exists on the Hub before merging."""
    repo_id = f"{username}/eka-{mode}-lora"
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        warn("huggingface_hub not installed — pip install -r ml/requirements.txt")
        return False

    api = HfApi(token=token)
    try:
        if hasattr(api, "repo_exists"):
            exists = api.repo_exists(repo_id, token=token)
        else:  # older huggingface_hub without repo_exists
            api.model_info(repo_id, token=token)
            exists = True
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 404:
            exists = False
        elif status == 401:
            fail(f"401 Unauthorized fetching {repo_id} — HF_TOKEN lacks access (is it gated?).")
            return False
        else:
            warn(f"Could not check {repo_id} ({exc}); assuming it exists and continuing.")
            return True
    except Exception as exc:
        warn(f"Could not check {repo_id} ({exc}); assuming it exists and continuing.")
        return True

    if not exists:
        fail(
            f"{repo_id} does not exist on the Hub.\n"
            f"        Train it first: python training/train_{mode}_lora_kaggle.py\n"
            f"        (or on Kaggle — see ml/notebooks/) then re-run this script."
        )
        return False
    ok(f"adapter found: {repo_id}")
    return True


def check_ollama_binary() -> Optional[str]:
    path = shutil.which("ollama")
    if not path:
        fail(
            "`ollama` is not on PATH.\n"
            "        Install it: https://ollama.com/download\n"
            "        Windows: winget install Ollama.Ollama\n"
            "        macOS:   brew install ollama\n"
            "        Linux:   curl -fsSL https://ollama.com/install.sh | sh"
        )
        return None
    ok(f"ollama binary: {path}")
    return path


def check_disk_space(n_modes_to_merge: int) -> bool:
    if n_modes_to_merge == 0:
        return True
    needed_gb = GB_PER_MODE * n_modes_to_merge
    free_gb = shutil.disk_usage(ML_DIR if ML_DIR.exists() else PROJECT_ROOT).free / 1e9
    if free_gb < needed_gb:
        warn(
            f"only {free_gb:.0f}GB free but merging {n_modes_to_merge} mode(s) "
            f"needs ~{needed_gb}GB. Free up disk or merge one mode at a time."
        )
        return False
    ok(f"{free_gb:.0f}GB free (need ~{needed_gb}GB for {n_modes_to_merge} mode(s))")
    return True


# ------------------------------------------------------------------- merging
def already_merged(mode: str) -> bool:
    out_dir = MERGED_DIR / mode
    if not (out_dir / "Modelfile").exists():
        return False
    has_weights = any(out_dir.glob("*.safetensors")) or (out_dir / "model.safetensors.index.json").exists()
    return has_weights


def do_merge(mode: str, dry_run: bool) -> bool:
    if already_merged(mode):
        ok(f"already merged at {MERGED_DIR / mode} — skipping merge step")
        return True

    if dry_run:
        print(f"    [dry-run] would run: {sys.executable} {MERGE_SCRIPT} --mode {mode}")
        return True

    if not MERGE_SCRIPT.exists():
        fail(f"merge script missing: {MERGE_SCRIPT}")
        return False

    step(f"merging adapter into base model for '{mode}' (this can take several minutes)")
    result = subprocess.run(
        [sys.executable, str(MERGE_SCRIPT), "--mode", mode],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        fail(f"merge_lora.py exited with code {result.returncode}")
        return False
    if not already_merged(mode):
        fail(f"merge_lora.py finished but {MERGED_DIR / mode} still has no Modelfile/weights")
        return False
    ok(f"merged -> {MERGED_DIR / mode}")
    return True


# ---------------------------------------------------------------------- gguf
def find_llama_cpp() -> Optional[Path]:
    candidates = []
    env_dir = os.environ.get("LLAMA_CPP_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(PROJECT_ROOT / "llama.cpp")
    candidates.append(PROJECT_ROOT.parent / "llama.cpp")
    candidates.append(Path.cwd() / "llama.cpp")

    for candidate in candidates:
        script = candidate / "convert_hf_to_gguf.py"
        if script.exists():
            return candidate
    return None


def convert_to_gguf(mode: str, dry_run: bool) -> bool:
    out_dir = MERGED_DIR / mode
    gguf_path = out_dir / f"{mode}.f16.gguf"
    if gguf_path.exists():
        ok(f"GGUF already exists: {gguf_path.name} — skipping conversion")
        rewrite_modelfile_from(out_dir / "Modelfile", gguf_path.name)
        return True

    llama_cpp_dir = find_llama_cpp()
    if not llama_cpp_dir:
        warn(
            "llama.cpp not found (checked $LLAMA_CPP_DIR, ./llama.cpp, ../llama.cpp).\n"
            "        GGUF conversion skipped — Ollama >= 0.1.32 imports safetensors\n"
            "        directly, so this is optional. To enable GGUF conversion later:\n\n"
            "          git clone https://github.com/ggerganov/llama.cpp\n"
            "          cd llama.cpp && pip install -r requirements.txt\n"
            "          (no build step needed for the Python conversion script)\n\n"
            "        then re-run this command with --gguf, or set LLAMA_CPP_DIR."
        )
        return False

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if dry_run:
        print(
            f"    [dry-run] would run: {sys.executable} {convert_script} {out_dir} "
            f"--outtype f16 --outfile {gguf_path}"
        )
        return True

    step(f"converting '{mode}' to GGUF (f16) via {llama_cpp_dir}")
    result = subprocess.run(
        [
            sys.executable, str(convert_script), str(out_dir),
            "--outtype", "f16",
            "--outfile", str(gguf_path),
        ],
        cwd=str(llama_cpp_dir),
    )
    if result.returncode != 0 or not gguf_path.exists():
        fail(f"GGUF conversion failed (exit {result.returncode})")
        return False

    rewrite_modelfile_from(out_dir / "Modelfile", gguf_path.name)
    ok(f"GGUF ready: {gguf_path}")
    return True


def rewrite_modelfile_from(modelfile_path: Path, gguf_filename: str) -> None:
    """Point the Modelfile's FROM line at the converted .gguf instead of ./"""
    if not modelfile_path.exists():
        warn(f"no Modelfile at {modelfile_path} to rewrite")
        return
    text = modelfile_path.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r"^FROM .*$", f"FROM ./{gguf_filename}", text, count=1, flags=re.MULTILINE
    )
    if count == 0:
        new_text = f"FROM ./{gguf_filename}\n" + text
    modelfile_path.write_text(new_text, encoding="utf-8")


# -------------------------------------------------------------- ollama create
def ollama_registered(mode: str) -> bool:
    model_name = f"eka-{mode}"
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return any(
        line.strip().split()[0].split(":")[0] == model_name
        for line in result.stdout.splitlines()[1:]
        if line.strip()
    )


def ollama_create(mode: str, dry_run: bool) -> bool:
    out_dir = MERGED_DIR / mode
    model_name = f"eka-{mode}"

    if ollama_registered(mode):
        ok(f"{model_name} already registered with Ollama — skipping create")
        return True

    if dry_run:
        print(f"    [dry-run] would run: ollama create {model_name} -f Modelfile  (cwd={out_dir})")
        return True

    if not (out_dir / "Modelfile").exists():
        fail(f"no Modelfile at {out_dir} — merge step must run first")
        return False

    step(f"registering {model_name} with Ollama")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", "Modelfile"],
        cwd=str(out_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail(f"`ollama create` failed:\n{result.stderr.strip()}")
        return False
    ok(f"{model_name} registered")
    return True


def ollama_verify(mode: str, dry_run: bool, timeout: float = 180.0) -> bool:
    model_name = f"eka-{mode}"
    if dry_run:
        print(f"    [dry-run] would run: ollama run {model_name} \"hello\"")
        return True

    step(f"verifying {model_name} responds")
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, "hello"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        fail(f"{model_name} did not respond within {timeout:.0f}s")
        return False

    output = (result.stdout or "").strip()
    if result.returncode != 0 or not output:
        fail(f"`ollama run {model_name}` produced no output. stderr: {(result.stderr or '').strip()[:300]}")
        return False

    preview = output[:200].replace("\n", " ")
    ok(f"response ({len(output)} chars): {preview}{'...' if len(output) > 200 else ''}")
    return True


# ------------------------------------------------------------------- one mode
def process_mode(mode: str, args: argparse.Namespace, creds: Dict[str, str]) -> Dict[str, object]:
    banner(f"MODE: {mode}")
    status = {
        "mode": mode, "merged": False, "gguf": False, "registered": False,
        "verified": False, "next_step": "",
    }

    if not args.skip_merge:
        if not check_adapter_on_hub(mode, creds["username"], creds["token"]):
            status["next_step"] = f"train it: python training/train_{mode}_lora_kaggle.py"
            return status
        if not do_merge(mode, args.dry_run):
            status["next_step"] = "fix the merge error above and re-run"
            return status
    else:
        if not already_merged(mode) and not args.dry_run:
            fail(f"--skip-merge passed but {MERGED_DIR / mode} has no merged model yet")
            status["next_step"] = "remove --skip-merge so this script merges it"
            return status
        ok("--skip-merge: assuming already merged")
    status["merged"] = True

    if args.gguf:
        status["gguf"] = convert_to_gguf(mode, args.dry_run)
        if not status["gguf"]:
            print("    continuing without GGUF (Ollama can import safetensors directly)")

    if not ollama_create(mode, args.dry_run):
        status["next_step"] = "fix the `ollama create` error above and re-run"
        return status
    status["registered"] = True

    if not ollama_verify(mode, args.dry_run):
        status["next_step"] = f"debug with: ollama run eka-{mode} \"hello\""
        return status
    status["verified"] = True
    status["next_step"] = "ready — set LLM_MODE=ollama (see below for remote access)"
    return status


# --------------------------------------------------------------------- ngrok
def start_ngrok(dry_run: bool) -> None:
    banner("NGROK TUNNEL")
    ngrok_bin = shutil.which("ngrok")
    if not ngrok_bin:
        fail(
            "ngrok is not on PATH.\n"
            "        Install: https://ngrok.com/download\n"
            "        Then:    ngrok config add-authtoken <your-token>  (from ngrok.com dashboard)"
        )
        return

    if dry_run:
        print(f"    [dry-run] would run: {ngrok_bin} http 11434")
        return

    step("starting `ngrok http 11434` in the background")
    try:
        subprocess.Popen(
            [ngrok_bin, "http", "11434"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        fail(f"could not start ngrok: {exc}")
        return

    url = None
    for _ in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen(NGROK_API, timeout=2) as response:
                data = json.loads(response.read().decode("utf-8"))
            for tunnel in data.get("tunnels", []):
                if tunnel.get("proto") == "https":
                    url = tunnel.get("public_url")
                    break
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        except Exception:
            pass
        if url:
            break

    if not url:
        fail(
            "ngrok did not report a tunnel within 30s.\n"
            "        Check http://127.0.0.1:4040 in a browser for the ngrok status page."
        )
        return

    print(f"\n    ngrok tunnel is live: {url}\n")
    print("    Paste these into the Render dashboard -> your service -> Environment:\n")
    print("        LLM_MODE=ollama")
    print(f"        OLLAMA_BASE_URL={url}\n")
    warn(
        "free ngrok URLs change every time the tunnel restarts (including your "
        "machine\n        rebooting or ngrok being killed) — you'll need to update "
        "OLLAMA_BASE_URL in\n        Render again after any restart. A paid ngrok "
        "reserved domain avoids this."
    )


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=MODES, help="single persona to merge+serve")
    parser.add_argument("--all", action="store_true", help="do all four personas")
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="adapter is already merged locally; skip straight to Ollama registration",
    )
    parser.add_argument(
        "--gguf", action="store_true",
        help="also convert the merged model to GGUF via a local llama.cpp checkout",
    )
    parser.add_argument(
        "--ngrok", action="store_true",
        help="after serving, start `ngrok http 11434` and print the public URL",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print every command that would run without executing side effects",
    )
    args = parser.parse_args()

    if not args.mode and not args.all:
        parser.error("pass --mode <founder|chanakya|gita|reflection> or --all")
    targets: List[str] = list(MODES) if args.all else [args.mode]

    banner("PHASE 7 — merge_lora_and_serve: prerequisites")
    creds = check_env_prereqs()
    if not creds:
        sys.exit(1)

    ollama_path = check_ollama_binary()
    if not ollama_path and not args.dry_run:
        sys.exit(1)

    to_merge = [m for m in targets if not args.skip_merge and not already_merged(m)]
    check_disk_space(len(to_merge))

    results = [process_mode(mode, args, creds) for mode in targets]

    if args.ngrok:
        start_ngrok(args.dry_run)

    banner("SUMMARY")
    print(f"{'mode':<11} {'merged':<8} {'registered':<12} {'verified':<10} next step")
    print("-" * 90)
    for r in results:
        print(
            f"{r['mode']:<11} "
            f"{'yes' if r['merged'] else 'no':<8} "
            f"{'yes' if r['registered'] else 'no':<12} "
            f"{'yes' if r['verified'] else 'no':<10} "
            f"{r['next_step']}"
        )

    all_verified = all(r["verified"] for r in results)
    print()
    if all_verified:
        print("All requested modes are merged, registered, and verified.")
        print("Next: set LLM_MODE=ollama (local backend) or run this again with --ngrok")
        print("      to expose Ollama to a remote Render deployment.")
    else:
        print("Some modes are not fully ready — see the 'next step' column above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
