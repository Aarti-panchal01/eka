"""Generate the Founder-mode training set: 1000 quality-gated pairs, 15 topics.

    python ml/scripts/generate_founder_data.py              # generate / resume
    python ml/scripts/generate_founder_data.py --report-only # re-score, no API calls
    python ml/scripts/generate_founder_data.py --plan        # estimate, no API calls

CPU only — this is pure Groq API calls. Resume-safe: kill it at any point and
re-run; state lives in founder_dataset.json and it picks up from there.

WHAT "QUALITY-GATED" MEANS HERE
------------------------------
Every pair must clear five gates (see _gen_quality.py) before it is written:
JSON shape, word-count floors, persona markers, structural rules, and a
diversity check against every pair already accepted. A rejected pair gets ONE
regeneration attempt with a prompt naming its specific defect, then is skipped.

At the end, founder_dataset_quality_report.json is written. If its verdict is
`needs_review`, do NOT train on the result — the blocking issues are listed in
the report and printed to the console.

backend/prompts/founder.txt was raised from a 200- to a 250-word cap so the
170-250 range below does not train the model to violate its own system prompt.
The runtime cap is settings.LLM_MAX_TOKENS = 512, which is ~385 words, so a
250-word response is not truncated at inference time.

RUNTIME AND THE TWO CEILINGS
----------------------------
Measured live against the Groq API on 2026-08-12 (not read from docs):

    llama-3.3-70b-versatile:  12,000 tokens/min   1,000 requests/DAY
    llama-3.1-8b-instant:      6,000 tokens/min  14,400 requests/DAY

Tokens per minute is the primary ceiling. Batching 5 pairs per call amortises
the ~1200-token persona prompt across five pairs (~2,350 -> ~830 tokens/pair),
which puts the ceiling around 700 pairs/hour on 70b. Expect 550-650 in practice
once rejections and regenerations are paid for, so this script is a ~1.5-2 hour
job for 1000 pairs.

Requests per day is the SECOND ceiling and it is the one to watch. 1000 pairs
is 200 batch calls, plus regenerations. That is comfortable alone, but all four
personas in one day (3,200 pairs = ~640 batch calls + ~160 regen calls) lands
near 70b's hard 1,000/day. If you run all four back to back and hit the cap,
finish the remainder with GROQ_GEN_MODEL=llama-3.1-8b-instant, which has
14,400/day — that is the situation the 8b model is actually useful for. It is
NOT faster: it has half the token ceiling.

Note: 8b-instant is a weaker teacher. The LoRA imitates whatever generated its
data, so prefer 70b for the bulk and treat 8b as the overflow valve.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gen_cli import run
from _gen_quality import PersonaSpec

MODE = "founder"
OUTPUT = "founder_dataset.json"
REPORT = "founder_dataset_quality_report.json"

# 15 topics summing to exactly 1000. The ten weighted to 70 are the situations
# a founder actually brings to Eka most often; the five at 60 are real but
# rarer. Quota is enforced with a +10% ceiling, so no topic can run away and
# skew the fine-tune toward one situation.
TOPIC_COUNTS = {
    "quit job to startup": 70,
    "co-founder conflict": 70,
    "fundraising rejection": 70,
    "product market fit doubt": 70,
    "startup failure and recovery": 70,
    "customer acquisition": 70,
    "founder burnout": 70,
    "scaling too fast": 70,
    "pricing strategy": 70,
    "pivot decision": 70,
    "hiring first employees": 60,
    "fear of competition": 60,
    "imposter syndrome": 60,
    "technical co-founder conflict": 60,
    "investor relationship": 60,
}  # = 1000  (10 x 70 + 5 x 60)

SPEC = PersonaSpec(
    mode=MODE,
    user_words=(100, 200),
    # The 170 floor sits just below the 180-word average target while the prompt
    # aims at the (170+250)/2 = 210 midpoint. A 150 floor let pairs land in the
    # 150-179 band and sink the average while every individual pair still
    # passed — measured: a 167-word response passed the gate and failed the
    # aggregate target. Raising the floor closes that gap.
    #
    # This is also why pairs left on disk from an earlier, looser run get
    # re-validated and dropped rather than inherited: a short pair would count
    # toward the total and drag avg_eka_length down while looking like progress.
    eka_words=(170, 250),
    # At least one of these must appear in the response. Single words match with
    # a word boundary plus suffixes, so "burn" catches "burn rate" and "burning"
    # but not an unrelated substring; "first-principles" is matched as a phrase.
    markers=(
        "framework",
        "assumption",
        "runway",
        "traction",
        "burn",
        "PMF",
        "first-principles",
    ),
    # The founder spec lists "specific question at end" as an acceptable marker
    # in its own right, so a response that closes on a real question satisfies
    # the gate even without one of the vocabulary markers above.
    question_counts_as_marker=True,
    # Founder mode is allowed to give advice — that is the whole persona. Only
    # reflection mode forbids it.
    forbid_advice=False,
    require_ends_with_question=False,
    require_verse=False,
    max_similarity=0.75,   # Jaccard; see DUPLICATE_THRESHOLD in _gen_quality.py
    min_marker_pass_rate=0.90,
    max_near_duplicates=0,  # binary guarantee: zero rows may duplicate an earlier row
    min_avg_eka_words=180,
)

EXTRA = (
    "The user should mention concrete details — revenue numbers, runway in "
    "months, team size, city — so Eka has something real to push back on. "
    "Eka should name the specific assumption or number that is actually load "
    "bearing, rather than offering general encouragement. "
)


if __name__ == "__main__":
    raise SystemExit(run(MODE, OUTPUT, REPORT, TOPIC_COUNTS, SPEC, EXTRA))
