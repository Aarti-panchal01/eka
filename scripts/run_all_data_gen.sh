#!/usr/bin/env bash
# =============================================================================
# EKA — one command that prepares everything for Kaggle training.
#
#   bash scripts/run_all_data_gen.sh
#
# Runs the full data pipeline sequentially, waiting for each stage to finish
# before starting the next. Every generation script is resume-safe, so if this
# dies (or you Ctrl-C it) just run it again — it picks up where it left off
# rather than regenerating from scratch.
#
# TOTAL RUNTIME: roughly 3.5-5 hours, almost all of it waiting on Groq's
# 30 req/min free-tier rate limit. Leave it running overnight.
#
#   founder     1000 pairs   ~45 min
#   chanakya     600 pairs   ~25 min
#   gita         630 pairs   ~27 min
#   reflection  1000 pairs   ~42 min
#   triplets    6000 rows    ~2.5 hrs   (one Groq call per anchor)
#   complexity  2000 rows    ~2 sec     (templates, no API)
#   preprocess               ~5 sec
#   upload                   ~30 sec
#
# ENV: needs GROQ_API_KEY. The upload stage also needs HF_TOKEN + HF_USERNAME.
#
# SKIPPING STAGES:  SKIP_TRIPLETS=1 bash scripts/run_all_data_gen.sh
#                   ONLY=founder,complexity bash scripts/run_all_data_gen.sh
# =============================================================================

set -uo pipefail   # deliberately NOT -e: a failed stage should be reported,
                   # not silently abort the remaining hours of work.

cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT="$(pwd)"
PY="${PY:-python}"

START_TS=$(date +%s)
FAILED=()
SKIPPED=()
COMPLETED=()

bar() { printf '=%.0s' {1..70}; printf '\n'; }

stage() {
    local key="$1" label="$2" script="$3"

    if [ -n "${ONLY:-}" ] && [[ ",$ONLY," != *",$key,"* ]]; then
        SKIPPED+=("$label (not in ONLY)")
        return 0
    fi
    local skipvar="SKIP_${key^^}"
    if [ "${!skipvar:-0}" = "1" ]; then
        SKIPPED+=("$label (${skipvar}=1)")
        printf '\n  ⏭  SKIP  %s\n' "$label"
        return 0
    fi

    bar
    printf '  %s\n' "$label"
    printf '  %s\n' "$PY $script"
    bar
    local t0
    t0=$(date +%s)

    if "$PY" "$script"; then
        local elapsed=$(( $(date +%s) - t0 ))
        printf '\n  ✅ done in %dm %ds\n\n' $((elapsed / 60)) $((elapsed % 60))
        COMPLETED+=("$label")
    else
        local code=$?
        printf '\n  ❌ FAILED (exit %d): %s\n' "$code" "$label"
        printf '     Continuing to the next stage. Re-run this script to retry;\n'
        printf '     the generators resume from whatever is already on disk.\n\n'
        FAILED+=("$label")
    fi
}

bar
echo "  EKA DATA GENERATION"
echo "  project: $PROJECT_ROOT"
echo "  started: $(date '+%Y-%m-%d %H:%M:%S')"
bar

# --- preflight ---------------------------------------------------------------
if [ -z "${GROQ_API_KEY:-}" ] && ! grep -qE '^GROQ_API_KEY=.+' .env 2>/dev/null; then
    echo "  ✗ GROQ_API_KEY not found in the environment or .env"
    echo "    Get a free key at console.groq.com and put it in .env"
    exit 1
fi
if ! "$PY" -c "import groq" 2>/dev/null; then
    echo "  ! groq SDK missing — installing"
    "$PY" -m pip install --quiet groq python-dotenv || {
        echo "  ✗ pip install failed"; exit 1; }
fi
echo "  ✓ preflight ok"
echo

# --- the pipeline ------------------------------------------------------------
stage founder    "1/8  Founder data (1000 pairs, ~45 min)"      ml/scripts/generate_founder_data.py
stage chanakya   "2/8  Chanakya data (600 pairs, ~25 min)"      ml/scripts/generate_chanakya_data.py
stage gita       "3/8  Gita data (630 pairs, ~27 min)"          ml/scripts/generate_gita_data.py
stage reflection "4/8  Reflection data (1000 pairs, ~42 min)"   ml/scripts/generate_reflection_data.py

# Triplets read all four datasets above, so this must come after them.
stage triplets   "5/8  Embedding triplets (6000, ~2.5 hrs)"     ml/scripts/generate_embedding_triplets.py
stage complexity "6/8  Complexity data (2000, no API)"          ml/scripts/generate_complexity_data.py
stage preprocess "7/8  Preprocess into Llama-3 splits"          ml/scripts/preprocess.py
stage upload     "8/8  Upload to HF Hub"                        ml/scripts/upload_to_hf.py

# --- summary -----------------------------------------------------------------
TOTAL=$(( $(date +%s) - START_TS ))
bar
printf '  SUMMARY — %dh %dm total\n' $((TOTAL / 3600)) $(((TOTAL % 3600) / 60))
bar
for s in "${COMPLETED[@]:-}"; do [ -n "$s" ] && printf '  ✅ %s\n' "$s"; done
for s in "${SKIPPED[@]:-}";   do [ -n "$s" ] && printf '  ⏭  %s\n' "$s"; done
for s in "${FAILED[@]:-}";    do [ -n "$s" ] && printf '  ❌ %s\n' "$s"; done
echo

echo "  Row counts on disk:"
for f in ml/datasets/*.json ml/datasets/*.jsonl; do
    [ -f "$f" ] || continue
    case "$f" in
        *.jsonl) n=$(grep -c . "$f" 2>/dev/null || echo 0) ;;
        *)       n=$("$PY" -c "import json,sys
try: print(len(json.load(open(sys.argv[1],encoding='utf-8'))))
except Exception: print('?')" "$f") ;;
    esac
    printf '    %-34s %s\n' "$(basename "$f")" "$n"
done
echo

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "  ⚠ ${#FAILED[@]} stage(s) failed — re-run this script to resume them."
    bar
    exit 1
fi

bar
echo "  === ALL DATA READY FOR TRAINING ==="
bar
cat <<'NEXT'

  Next, on Kaggle (one notebook per persona, ~2-3 hrs each):

    1. kaggle.com -> Create -> Notebook -> File -> Upload
       training/train_founder_lora_kaggle.py
    2. Add-ons -> Secrets:  HF_TOKEN, WANDB_API_KEY, HF_USERNAME
    3. Settings: Accelerator = GPU T4, Internet = ON
    4. Save Version -> "Save & Run All (Commit)"   <- background execution
    5. Repeat for chanakya / gita / reflection, then:
       train_embeddings_kaggle.py, train_classifiers_kaggle.py

  Locally, right now (CPU, under 10 minutes):

    python training/train_ranker_local.py

  Check progress any time:

    python scripts/check_build_status.py

NEXT
