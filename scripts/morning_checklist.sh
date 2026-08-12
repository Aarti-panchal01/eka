#!/usr/bin/env bash
# Eka — morning status. Read-only: no API calls, no generation, no uploads.
#
# Run this first thing:  bash scripts/morning_checklist.sh
#
# Answers, in order: did anything finish overnight, is anything still running,
# which providers came back, and is the data good enough to train on. Every
# check is local except the optional provider probe at the end.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || PY="/c/Users/Aarti Panchal/AppData/Local/Programs/Python/Python313/python.exe"

line() { printf '%s\n' "----------------------------------------------------------------"; }
head2() { printf '\n%s\n' "== $* =="; }

printf '\n================================================================\n'
printf '  EKA — morning status   %s\n' "$(date '+%Y-%m-%d %H:%M')"
printf '================================================================\n'

# ---------------------------------------------------------------- 1. datasets
head2 "1. Dataset progress"
"$PY" ml/scripts/run_queue.py --status 2>/dev/null || \
  echo "  (run_queue --status failed — check the traceback by running it directly)"

# ------------------------------------------------------------- 2. still alive
head2 "2. Anything still running?"
if command -v tasklist >/dev/null 2>&1; then
  # tasklist has no command-line column, so this is a coarse signal only.
  n=$(tasklist /FI "IMAGENAME eq python.exe" /NH 2>/dev/null | grep -ci "python.exe" || true)
  if [ "${n:-0}" -gt 0 ]; then
    echo "  $n python process(es) alive — generation may still be going."
    echo "  Confirm with the log tail in section 3 before restarting anything."
  else
    echo "  No python processes — generation has stopped."
    echo "  Resume with:  $PY ml/scripts/run_queue.py --all"
  fi
else
  pgrep -fa "run_queue.py|generate_.*_data.py" 2>/dev/null || echo "  none running"
fi

# ------------------------------------------------------------------- 3. logs
head2 "3. Last activity per run"
for log in ml/datasets/queue_run.log ml/datasets/founder_run.log \
           ml/datasets/chanakya_run.log ml/datasets/gita_run.log \
           ml/datasets/reflection_run.log; do
  [ -f "$log" ] || continue
  printf '\n  %s\n' "$log"
  # Last real progress line, plus any stop reason.
  grep -E '^✓|^↻' "$log" 2>/dev/null | tail -2 | sed 's/^/    /'
  grep -E 'STOPPING|out of quota|exhausted|QUEUE SUMMARY' "$log" 2>/dev/null | tail -2 | sed 's/^/    /'
done

# -------------------------------------------------------------- 4. throughput
head2 "4. Overnight throughput (from queue log)"
"$PY" - <<'PYEOF' 2>/dev/null || echo "  (could not compute)"
import re, os, time, pathlib
log = pathlib.Path("ml/datasets/queue_run.log")
if not log.exists():
    print("  no queue log yet"); raise SystemExit
text = log.read_text(encoding="utf-8", errors="replace")
kept = sum(int(m) for m in re.findall(r"\+(\d+) kept", text))
rej  = sum(int(m) for m in re.findall(r"kept, (\d+) rejected", text))
# mtime-mtime is unreliable on Windows (file tunneling preserves ctime), so
# report totals and let the operator judge the window rather than inventing a rate.
age = (time.time() - log.stat().st_mtime) / 60
print(f"  kept={kept}  rejected={rej}" + (f"  accept={kept/(kept+rej):.0%}" if kept+rej else ""))
print(f"  log last written {age:.0f} min ago")
if age > 15:
    print("  -> stale: the run has almost certainly stopped")
PYEOF

# ------------------------------------------------------------ 5. quality gate
head2 "5. Quality verdicts (is it trainable?)"
"$PY" - <<'PYEOF' 2>/dev/null || echo "  (could not read reports)"
import json, pathlib
d = pathlib.Path("ml/datasets")
any_report = False
for name in ("founder", "chanakya", "gita", "reflection"):
    p = d / f"{name}_dataset_quality_report.json"
    if not p.exists():
        print(f"  {name:11} no report yet")
        continue
    any_report = True
    r = json.loads(p.read_text(encoding="utf-8"))
    mark = "OK " if r.get("verdict") == "ok_to_train" else "!! "
    print(f"  {mark}{name:11} {r.get('verdict','?'):14} pairs={r.get('total_pairs')} "
          f"marker={r.get('persona_marker_pass_rate')} unique={r.get('unique_pair_guarantee')}")
    if r.get("verdict") != "ok_to_train":
        for issue in (r.get("blocking_issues") or [])[:3]:
            print(f"        - {issue}")
if not any_report:
    print("  (none yet — reports are written when a persona run completes)")
PYEOF

# ------------------------------------------------------------------ 6. splits
head2 "6. Splits / Hub readiness"
if [ -d ml/data/splits ]; then
  ls -1 ml/data/splits/*.jsonl 2>/dev/null | while read -r f; do
    printf '  %-46s %s lines\n' "$(basename "$f")" "$(wc -l < "$f" | tr -d ' ')"
  done
  [ -z "$(ls -A ml/data/splits 2>/dev/null)" ] && echo "  (empty — run preprocess.py)"
else
  echo "  no splits yet — run:  $PY ml/scripts/preprocess.py"
fi

# ------------------------------------------------------------------- 7. git
head2 "7. Git"
if [ -d .git ]; then
  echo "  commits:  $(git rev-list --count HEAD 2>/dev/null || echo 0)"
  echo "  remote:   $(git remote -v 2>/dev/null | head -1 || echo '(none — see TOMORROW.md)')"
  echo "  modified: $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') file(s)"
  # The one check that must never fail: secrets stay out of git.
  if [ "$(git ls-files 2>/dev/null | grep -c '^\.env$')" != "0" ]; then
    echo "  !! .env IS TRACKED BY GIT — remove it before any push"
  else
    echo "  .env untracked: ok"
  fi
else
  echo "  not a git repo"
fi

# --------------------------------------------------------------- 8. next step
head2 "8. What to do now"
"$PY" - <<'PYEOF' 2>/dev/null
import json, pathlib, re
d = pathlib.Path("ml/datasets")
s = pathlib.Path("ml/scripts")
def target(module):
    t, inside = 0, False
    for line in (s / module).read_text(encoding="utf-8").splitlines():
        if line.startswith("TOPIC_COUNTS"): inside = True; continue
        if inside:
            if line.startswith("}"): break
            if ":" in line:
                tail = line.rsplit(":", 1)[1].strip().rstrip(",")
                if tail.isdigit(): t += int(tail)
    return t
def have(ds):
    try: return len(json.loads((d / ds).read_text(encoding="utf-8")))
    except Exception: return 0
P = [("founder","generate_founder_data.py","founder_dataset.json"),
     ("chanakya","generate_chanakya_data.py","chanakya_dataset.json"),
     ("gita","generate_gita_data.py","gita_dataset.json"),
     ("reflection","generate_reflection_data.py","reflection_dataset.json")]
short = [(n, target(m) - have(ds)) for n, m, ds in P if have(ds) < target(m)]
if not short:
    print("  All four at target. Publish and start Kaggle:")
    print("    python scripts/watch_and_publish.py")
else:
    total = sum(n for _, n in short)
    print(f"  {total} pairs still needed: " + ", ".join(f"{n} ({k})" for n, k in short))
    print("  1) check which providers came back:")
    print("       python ml/scripts/_gen_providers.py --check")
    print("  2) resume generation (resume-safe, re-runnable):")
    print("       python ml/scripts/run_queue.py --all")
    print("  3) leave the watcher running so publishing happens unattended:")
    print("       python scripts/watch_and_publish.py")
    print("  Reminder: 76-216 pairs/hr measured. Re-measure before trusting an ETA.")
    print("  The Anthropic Batches API does the remainder for ~$4.50 (see TOMORROW.md).")
PYEOF

printf '\n  Provider health (one real call each, ~40 tokens):\n'
printf '    %s ml/scripts/_gen_providers.py --check\n' "$PY"
printf '  Full plan: TOMORROW.md\n\n'
