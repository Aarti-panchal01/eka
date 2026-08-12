"""EKA build status dashboard — "what's done and what's left", in one command.

    python scripts/check_build_status.py

Reads local files, the Hugging Face Hub, and (if it's up) the live backend,
then prints a single report: models, data, services, and a concrete list of
next commands derived from whatever is actually missing. Never raises and
never exits non-zero — this is a report, not a test.

Dependencies: stdlib only, +httpx +huggingface_hub +python-dotenv if present.
Every import is optional and degrades with a plain message when absent.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the box/emoji
# characters used below (✅ ⏳ ⬜ ⚠ █ ░ …). Do this before any printing.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ML_DIR = PROJECT_ROOT / "ml"
DATASETS_DIR = ML_DIR / "datasets"
SPLITS_DIR = ML_DIR / "data" / "splits"
RANKER_DIR = ML_DIR / "models" / "ranker"

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

try:
    import httpx
except ImportError:
    httpx = None

try:
    from huggingface_hub import HfApi

    HF_AVAILABLE = True
except ImportError:
    HfApi = None
    HF_AVAILABLE = False

HF_USERNAME = os.environ.get("HF_USERNAME", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
EKA_BASE_URL = os.environ.get("EKA_BASE_URL", "http://localhost:8000").rstrip("/")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL_PREFIX = os.environ.get("OLLAMA_MODEL_PREFIX", "eka").strip()

WIDTH = 78

# Facts collected as we go, used to build the NEXT ACTIONS section at the end.
facts = {
    "hf_username_set": bool(HF_USERNAME),
    "hf_repos": {},          # repo -> "found" | "missing" | "unauthorized" | "unknown"
    "datasets": {},          # filename -> {"count": int, "target": int}
    "splits_present": False,
    "ranker_exists": False,
    "feature_names_exists": False,
    "backend_reachable": False,
    "backend_status": None,
    "standalone": {},        # port -> True/False/None(not installed)
    "ollama_models": [],
}


# ============================================================== formatting
def header(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f" {title}")
    print("=" * WIDTH)


def bar(pct: float, width: int = 22) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    return "█" * filled + "░" * (width - filled)


def human_ago(dt) -> str:
    if dt is None:
        return "unknown time"
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        seconds = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return "unknown time"

    if seconds < 60:
        return f"{int(seconds)}s ago"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)}m ago"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)}h ago"
    days = hours / 24
    if days < 30:
        return f"{int(days)}d ago"
    months = days / 30
    if months < 12:
        return f"{int(months)}mo ago"
    return f"{int(months / 12)}y ago"


# ================================================================== MODELS
def section_models() -> None:
    header("MODELS (Hugging Face Hub)")

    if not HF_USERNAME:
        print(
            "⏳ HF_USERNAME is empty in .env — nothing on the Hub can be checked "
            "until it's set. Every repo below hangs off it (see backend/config.py: "
            "hub_repo())."
        )
        return

    if not HF_AVAILABLE:
        print("⚠ huggingface_hub is not installed — pip install huggingface-hub")
        return

    model_repos = [
        "eka-founder-lora",
        "eka-chanakya-lora",
        "eka-gita-lora",
        "eka-reflection-lora",
        "eka-embeddings",
        "eka-complexity",
        "eka-sentiment",
        "eka-summarizer",
    ]
    dataset_repos = ["eka-datasets"]

    api = HfApi(token=HF_TOKEN or None)

    for repo, repo_type in [(r, "model") for r in model_repos] + [
        (r, "dataset") for r in dataset_repos
    ]:
        repo_id = f"{HF_USERNAME}/{repo}"
        try:
            if repo_type == "dataset":
                info = api.dataset_info(repo_id, token=HF_TOKEN or None)
            else:
                info = api.model_info(repo_id, token=HF_TOKEN or None)
            n_files = len(info.siblings) if info.siblings else 0
            last_mod = getattr(info, "last_modified", None) or getattr(
                info, "lastModified", None
            )
            print(f"✅ {repo:<22} {n_files:>3} files   pushed {human_ago(last_mod)}")
            facts["hf_repos"][repo] = "found"
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                print(f"❌ {repo:<22} 401 unauthorized — check HF_TOKEN in .env")
                facts["hf_repos"][repo] = "unauthorized"
            elif status == 404 or "NotFound" in type(exc).__name__:
                print(f"⏳ {repo:<22} not yet — https://huggingface.co/{repo_id}")
                facts["hf_repos"][repo] = "missing"
            else:
                print(f"⚠ {repo:<22} {type(exc).__name__}: {exc}")
                facts["hf_repos"][repo] = "unknown"


# ==================================================================== DATA
def _count_json_array(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else -1
    except Exception:
        return -1


def _count_jsonl(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
        return -1


def section_data() -> None:
    header("DATA (local)")

    print(" ml/datasets/")
    targets = [
        ("founder_dataset.json", 1000, "json"),
        ("chanakya_dataset.json", 600, "json"),
        ("gita_dataset.json", 630, "json"),
        ("reflection_dataset.json", 1000, "json"),
        ("embedding_triplets.jsonl", 6000, "jsonl"),
        ("complexity_labeled.jsonl", 2000, "jsonl"),
    ]
    for name, target, kind in targets:
        path = DATASETS_DIR / name
        if not path.exists():
            count = 0
        elif kind == "json":
            count = _count_json_array(path)
        else:
            count = _count_jsonl(path)

        facts["datasets"][name] = {"count": max(count, 0), "target": target}

        if count < 0:
            print(f"  ⚠ {name:<28} unreadable / bad format")
            continue

        pct = 100.0 * count / target if target else 0.0
        icon = "✅" if count >= target else ("⏳" if count > 0 else "⬜")
        print(
            f"  {icon} {name:<28} {count:>5}/{target:<5} {pct:>5.1f}%  {bar(pct)}"
        )

    print("\n ml/data/splits/")
    any_split = False
    for mode in ("founder", "chanakya", "gita", "reflection"):
        for split in ("train", "val"):
            path = SPLITS_DIR / f"{mode}_{split}.jsonl"
            if path.exists():
                n = _count_jsonl(path)
                any_split = True
                print(f"  ✅ {path.name:<28} {n:>5} lines")
            else:
                print(f"  ⬜ {path.name:<28} missing")
    facts["splits_present"] = any_split

    print("\n ml/models/ranker/")
    ranker_path = RANKER_DIR / "eka_ranker.txt"
    features_path = RANKER_DIR / "feature_names.json"

    if ranker_path.exists():
        print(f"  ✅ {ranker_path.name} exists")
        facts["ranker_exists"] = True
    else:
        print(f"  ⬜ {ranker_path.name} missing — run: python training/train_ranker_local.py")

    if features_path.exists():
        facts["feature_names_exists"] = True
        try:
            meta = json.loads(features_path.read_text(encoding="utf-8"))
            ndcg3 = meta.get("ndcg_at_3")
            if ndcg3 is not None:
                target_note = "✅ above 0.70 target" if ndcg3 >= 0.70 else "⚠ below 0.70 target"
                print(f"  ✅ {features_path.name} exists — ndcg_at_3 = {ndcg3:.4f} ({target_note})")
            else:
                print(f"  ✅ {features_path.name} exists (no ndcg_at_3 recorded)")
        except Exception as exc:
            print(f"  ⚠ {features_path.name} unreadable: {exc}")
    else:
        print(f"  ⬜ {features_path.name} missing")


# ================================================================ SERVICES
def _get_json(url, timeout=3.0):
    if httpx is None:
        return None, "httpx not installed"
    try:
        response = httpx.get(url, timeout=timeout)
        return response, None
    except Exception as exc:
        return None, str(exc)


def _post_json(url, payload, timeout=3.0):
    if httpx is None:
        return None, "httpx not installed"
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        return response, None
    except Exception as exc:
        return None, str(exc)


def _flag(value) -> str:
    return "✅" if value else "❌"


def section_services() -> None:
    header("SERVICES (live)")

    if httpx is None:
        print("⚠ httpx is not installed — pip install httpx. Skipping all live checks.")
        return

    print(f" backend  GET {EKA_BASE_URL}/health")
    response, error = _get_json(f"{EKA_BASE_URL}/health")

    if response is not None and response.status_code == 200:
        try:
            body = response.json()
        except Exception:
            body = {}
        facts["backend_reachable"] = True
        facts["backend_status"] = body.get("status")

        print(f"  overall status: {body.get('status', 'unknown')}")
        for field in (
            "ollama", "groq", "hf_space", "qdrant", "database",
            "complexity", "ranker", "sentiment", "sarvam_configured",
        ):
            print(f"  {_flag(body.get(field))} {field}")

        details = body.get("details") or {}
        vectors = details.get("vectors") or {}
        print(
            f"  embedding tier: {vectors.get('embedding_tier', 'unknown')}   "
            f"qdrant points: {vectors.get('points', 'unknown')}"
        )
        return

    facts["backend_reachable"] = False
    print(f"  ❌ unreachable — {error or (response and response.status_code)}")
    print(f"  start it with: make serve-local  (or: uvicorn main:app --reload --app-dir backend)")

    # Backend is down — probe the optional standalone services directly so
    # the report still says something useful about what's running.
    print("\n standalone services (optional, direct probes since backend is down)")

    resp, err = _post_json(
        "http://localhost:8004/classify",
        {"text": "How do I balance growth with my own health while raising a seed round?"},
    )
    if resp is not None and resp.status_code == 200:
        body = resp.json()
        print(f"  ✅ complexity :8004  label={body.get('label')} tier={body.get('tier')}")
        facts["standalone"]["complexity"] = True
    else:
        print(f"  ⏳ complexity :8004  not running — python serving/complexity_serve.py")
        facts["standalone"]["complexity"] = False

    resp, err = _post_json(
        "http://localhost:8005/rank",
        {"features": [[0.82, 7.0, 2.0, 15.0, 3.0, 1.0, 1.0, 0.35]]},
    )
    if resp is not None and resp.status_code == 200:
        body = resp.json()
        print(f"  ✅ ranker     :8005  scores={body.get('scores')} tier={body.get('tier')}")
        facts["standalone"]["ranker"] = True
    else:
        print(f"  ⏳ ranker     :8005  not running — python serving/ranker_serve.py")
        facts["standalone"]["ranker"] = False

    resp, err = _post_json(
        "http://localhost:8007/sentiment",
        {"text": "I'm anxious about the fundraising call tomorrow."},
    )
    if resp is not None and resp.status_code == 200:
        body = resp.json()
        print(f"  ✅ sentiment  :8007  label={body.get('label')} tier={body.get('tier')}")
        facts["standalone"]["sentiment"] = True
    else:
        print(f"  ⏳ sentiment  :8007  not running — python serving/sentiment_serve.py")
        facts["standalone"]["sentiment"] = False

    resp, err = _get_json(f"{OLLAMA_BASE_URL}/api/tags")
    if resp is not None and resp.status_code == 200:
        try:
            models = resp.json().get("models", [])
        except Exception:
            models = []
        names = [m.get("name", "") for m in models]
        eka_names = [n for n in names if n.startswith(f"{OLLAMA_MODEL_PREFIX}-")]
        facts["ollama_models"] = eka_names
        if eka_names:
            print(f"  ✅ ollama     {OLLAMA_BASE_URL}  {', '.join(eka_names)}")
        else:
            print(f"  ⏳ ollama     {OLLAMA_BASE_URL}  reachable, but no {OLLAMA_MODEL_PREFIX}-* models yet")
    else:
        print(f"  ⏳ ollama     {OLLAMA_BASE_URL}  not reachable — {err or 'connection failed'}")


# ============================================================ NEXT ACTIONS
def section_next_actions() -> None:
    header("NEXT ACTIONS")

    actions = []

    if not facts["hf_username_set"]:
        actions.append(
            "Set HF_USERNAME in .env (huggingface.co/<username>) — every model "
            "repo and the Hub dataset repo hang off it. Nothing HF-related can "
            "be checked or trained until this is set."
        )

    persona_targets = [
        ("founder_dataset.json", "generate_founder_data.py"),
        ("chanakya_dataset.json", "generate_chanakya_data.py"),
        ("gita_dataset.json", "generate_gita_data.py"),
        ("reflection_dataset.json", "generate_reflection_data.py"),
    ]
    short_personas = [
        script
        for name, script in persona_targets
        if facts["datasets"].get(name, {}).get("count", 0)
        < facts["datasets"].get(name, {}).get("target", 1)
    ]
    if len(short_personas) == 4:
        actions.append(
            "Generate the four persona datasets (Groq API, ~45min each): make data-all"
        )
    elif short_personas:
        for script in short_personas:
            actions.append(f"Finish this persona dataset: python ml/scripts/{script}")

    triplets = facts["datasets"].get("embedding_triplets.jsonl", {})
    if triplets.get("count", 0) < triplets.get("target", 1):
        if short_personas:
            actions.append(
                "Once persona datasets exist, build embedding triplets: "
                "python ml/scripts/generate_embedding_triplets.py"
            )
        else:
            actions.append("python ml/scripts/generate_embedding_triplets.py")

    complexity = facts["datasets"].get("complexity_labeled.jsonl", {})
    if complexity.get("count", 0) < complexity.get("target", 1):
        actions.append("python ml/scripts/generate_complexity_data.py  (no API needed)")

    all_persona_ready = all(
        facts["datasets"].get(name, {}).get("count", 0)
        >= facts["datasets"].get(name, {}).get("target", 1)
        for name, _ in persona_targets
    )
    if all_persona_ready and not facts["splits_present"]:
        actions.append("Build train/val splits: python ml/scripts/preprocess.py")

    if facts["splits_present"] and facts["hf_username_set"]:
        lora_repos = [
            r for r in ("eka-founder-lora", "eka-chanakya-lora", "eka-gita-lora", "eka-reflection-lora")
        ]
        any_lora_missing = any(facts["hf_repos"].get(r) != "found" for r in lora_repos)
        if any_lora_missing:
            actions.append(
                "Push data to the Hub, then train on Kaggle: python ml/scripts/upload_to_hf.py, "
                "then upload training/train_founder_lora_kaggle.py (etc.) as Kaggle notebooks "
                "with HF_TOKEN/HF_USERNAME/WANDB_API_KEY as secrets, background execution on"
            )
    elif facts["splits_present"] and not facts["hf_username_set"]:
        actions.append(
            "Splits are ready but HF_USERNAME is empty — set it, then: "
            "python ml/scripts/upload_to_hf.py"
        )

    if not facts["ranker_exists"]:
        actions.append("Train the local reranker: python training/train_ranker_local.py")

    if not facts["backend_reachable"]:
        actions.append("Bring the backend up for a live /health read: make serve-local")

    if not actions:
        actions.append("Everything checked out — nothing obviously missing.")

    for i, action in enumerate(actions[:6], start=1):
        print(f"  {i}. {action}")


def main() -> int:
    print("=" * WIDTH)
    print(" EKA BUILD STATUS".center(WIDTH))
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(WIDTH))
    print("=" * WIDTH)

    for section in (section_models, section_data, section_services, section_next_actions):
        try:
            section()
        except Exception as exc:
            print(f"\n⚠ section {section.__name__} crashed: {type(exc).__name__}: {exc}")

    print()
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n⚠ dashboard crashed: {type(exc).__name__}: {exc}")
    sys.exit(0)
