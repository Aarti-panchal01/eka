#!/usr/bin/env python3
"""Check every credential in .env actually works, after a rotation.

    python scripts/verify_credentials.py

Rotating eight keys by hand across six dashboards is exactly the kind of job
where one gets missed, and the way you find out is a run dying three hours in.
This makes a real call against each one and prints a pass/fail table.

Read-only: nothing here writes, uploads, or spends meaningful quota.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
except ImportError:
    print("! python-dotenv missing — reading real environment only\n")

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<22} {detail}")


def get(key: str) -> str:
    return (os.environ.get(key) or "").strip()


def http(url: str, headers: dict, timeout: int = 25) -> tuple[int, str]:
    # A User-Agent is not optional. Without one, Groq's edge returns a
    # Cloudflare 403 "error code: 1010" that looks exactly like a dead key —
    # the first version of this script reported a perfectly good key as FAIL.
    headers = {"User-Agent": "eka-credential-check/1.0", **headers}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(400).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(300).decode("utf-8", "replace")
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"


print("Credential check\n")

# ---------------------------------------------------------------- hugging face
token = get("HF_TOKEN")
if not token:
    record("HF_TOKEN", False, "not set")
else:
    # Read 400 bytes and the JSON is truncated mid-object, so the name lookup
    # silently returned "?" and failed a working token. Authentication is what
    # the 200 proves; the username is a nice-to-have, not the test.
    code, body = http("https://huggingface.co/api/whoami-v2",
                      {"Authorization": f"Bearer {token}"}, timeout=25)
    if code == 200:
        who = ""
        try:
            who = json.loads(body).get("name", "")
        except Exception:
            pass
        expected = get("HF_USERNAME") or "amijackofalltrades"
        if who and who != expected:
            record("HF_TOKEN", False, f"authenticates as {who} — expected {expected}")
        else:
            record("HF_TOKEN", True, f"authenticates as {who or 'ok'}")
    else:
        record("HF_TOKEN", False, f"HTTP {code}")

# ------------------------------------------------------------------ openai-ish
# Every one of these speaks the OpenAI /models shape, so one probe covers them.
OPENAI_LIKE = [
    ("GROQ_API_KEY", "https://api.groq.com/openai/v1/models"),
    ("MISTRAL_API_KEY", "https://api.mistral.ai/v1/models"),
    ("MISTRAL_API_KEY_2", "https://api.mistral.ai/v1/models"),
    ("MISTRAL_API_KEY_3", "https://api.mistral.ai/v1/models"),
    ("MISTRAL_API_KEY_4", "https://api.mistral.ai/v1/models"),
    ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/models"),
    ("SAMBANOVA_API_KEY", "https://api.sambanova.ai/v1/models"),
]
for key, url in OPENAI_LIKE:
    val = get(key)
    if not val:
        record(key, False, "not set")
        continue
    code, body = http(url, {"Authorization": f"Bearer {val}"})
    record(key, code == 200, f"HTTP {code}" + ("" if code == 200 else f" {body[:70]}"))

# ---------------------------------------------------------------------- google
val = get("GOOGLE_API_KEY")
if not val:
    record("GOOGLE_API_KEY", False, "not set")
else:
    code, body = http(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={val}", {})
    record("GOOGLE_API_KEY", code == 200,
           f"HTTP {code}" + ("" if code == 200 else f" {body[:70]}"))

# ---------------------------------------------------------------------- github
val = get("GITHUB_TOKEN")
if not val:
    record("GITHUB_TOKEN", True, "not set (fine — GitHub Models is retired)")
else:
    code, _ = http("https://api.github.com/user",
                   {"Authorization": f"Bearer {val}",
                    "Accept": "application/vnd.github+json"})
    record("GITHUB_TOKEN", code == 200, f"HTTP {code}")

# ---------------------------------------------------------------------- kaggle
kj = Path(os.path.expanduser("~")) / ".kaggle" / "kaggle.json"
if not kj.exists():
    record("kaggle.json", False, "missing")
else:
    # Shell out to the CLI rather than hand-rolling the REST call. The raw
    # endpoint wants pagination params and returned 400 for a valid key, which
    # is a false alarm about the one credential the orchestrator depends on.
    import subprocess

    try:
        cfg = json.loads(kj.read_text(encoding="utf-8"))
        r = subprocess.run(
            ["kaggle", "kernels", "list", "-m"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        out = (r.stdout or "") + (r.stderr or "")
        ok = "ref" in out and "401" not in out and "403" not in out
        record("kaggle.json", ok, f"{cfg.get('username', '?')} — "
               + ("CLI authenticates" if ok else out.strip()[:70]))
    except Exception as exc:
        record("kaggle.json", False, str(exc)[:70])

# ------------------------------------------------------------------- inert-ish
for key in ("QDRANT_API_KEY", "SARVAM_API_KEY", "WANDB_API_KEY",
            "DATABASE_URL", "SECRET_KEY"):
    record(key, bool(get(key)), "present" if get(key) else "not set — check .env")

failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    print("FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
