"""Merge a trained LoRA adapter into the base Llama-3 and prep it for Ollama.

    python ml/scripts/merge_lora.py --mode founder
    python ml/scripts/merge_lora.py --all

Needs ~16GB RAM (fp16 8B model) and ~16GB disk per mode. Runs on CPU — slow
but it works without a GPU. Run this wherever Ollama will live.

# DEPLOY_LATER: run this once the LoRA adapters exist on HF Hub, then switch
# LLM_MODE=ollama in .env (or point OLLAMA_BASE_URL at an ngrok tunnel).
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode ✓/✅/⚠.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPTS_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = ML_DIR.parent
MERGED_DIR = ML_DIR / "models" / "merged"

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

MODES = ("founder", "chanakya", "gita", "reflection")
BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

MODELFILE = """FROM ./
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "User:"

TEMPLATE \"\"\"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

\"\"\"

SYSTEM \"\"\"{system}\"\"\"
"""


def merge_one(mode: str, username: str, token: str, dtype_name: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_repo = f"{username}/eka-{mode}-lora"
    out_dir = MERGED_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]

    print(f"\n=== {mode} ===")
    print(f"  base    : {BASE_MODEL}")
    print(f"  adapter : {adapter_repo}")
    print(f"  output  : {out_dir}")

    print("  loading base model (this takes a few minutes and ~16GB RAM)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)

    print("  applying adapter...")
    model = PeftModel.from_pretrained(base, adapter_repo, token=token)

    print("  merging weights...")
    model = model.merge_and_unload()

    print("  saving merged model...")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    # Bake the persona in as the Ollama SYSTEM prompt so `ollama run eka-founder`
    # behaves correctly even without the backend supplying it.
    persona_path = PROJECT_ROOT / "backend" / "prompts" / f"{mode}.txt"
    persona = persona_path.read_text(encoding="utf-8").strip() if persona_path.exists() else ""
    (out_dir / "Modelfile").write_text(MODELFILE.format(system=persona), encoding="utf-8")

    del model, base
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, help="single mode to merge")
    parser.add_argument("--all", action="store_true", help="merge all four")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    args = parser.parse_args()

    if not args.mode and not args.all:
        parser.error("pass --mode <name> or --all")

    token = os.environ.get("HF_TOKEN", "").strip()
    username = os.environ.get("HF_USERNAME", "").strip()
    if not token or not username:
        sys.exit("HF_TOKEN and HF_USERNAME must be set in .env")

    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        sys.exit("pip install -r ml/requirements.txt")

    free_gb = shutil.disk_usage(MERGED_DIR.parent).free / 1e9
    targets = list(MODES) if args.all else [args.mode]
    if free_gb < 16 * len(targets):
        print(
            f"⚠ only {free_gb:.0f}GB free — each merged model is ~16GB "
            f"({len(targets)} requested)"
        )

    done = []
    for mode in targets:
        try:
            done.append(merge_one(mode, username, token, args.dtype))
        except Exception as exc:
            print(f"✗ {mode} failed: {exc}")

    if not done:
        sys.exit(1)

    print("\n✅ merged models ready. Register them with Ollama:\n")
    for path in done:
        mode = path.name
        print(f"  cd {path} && ollama create eka-{mode} -f Modelfile")
    print("\nVerify:")
    print(f"  ollama run eka-{done[0].name} \"hello\"")
    print("\nExpose to a remote backend (Render) with ngrok:")
    print("  ngrok http 11434")
    print("  -> set OLLAMA_BASE_URL=<ngrok url> and LLM_MODE=ollama in Render env\n")
    print(
        "NOTE: `ollama create` imports safetensors directly on Ollama >= 0.1.32.\n"
        "On older versions convert to GGUF first:\n"
        "  python llama.cpp/convert_hf_to_gguf.py <merged_dir> --outtype f16\n"
        "  then change the Modelfile's FROM line to the .gguf path."
    )


if __name__ == "__main__":
    main()
