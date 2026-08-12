"""Generate the Chanakya-mode training set: 600 quality-gated pairs, 10 topics.

    python ml/scripts/generate_chanakya_data.py              # generate / resume
    python ml/scripts/generate_chanakya_data.py --report-only # re-score, no API calls
    python ml/scripts/generate_chanakya_data.py --plan        # estimate, no API calls

Resume-safe, CPU only. See _gen_cli.py for the shared CLI and _gen_quality.py
for the five gates every pair must clear before it is written.

backend/prompts/chanakya.txt was raised from a 200- to a 250-word cap so the
160-250 range below does not train the model to violate its own system prompt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gen_cli import run
from _gen_quality import PersonaSpec

MODE = "chanakya"
OUTPUT = "chanakya_dataset.json"
REPORT = "chanakya_dataset_quality_report.json"

TOPIC_COUNTS = {
    "business betrayal": 60,
    "power dynamics at work": 60,
    "alliance building": 60,
    "recognizing hidden enemies": 60,
    "timing a major decision": 60,
    "negotiation strategy": 60,
    "managing ego in conflict": 60,
    "resource control": 60,
    "political maneuvering": 60,
    "revenge vs strategy": 60,
}  # = 600

SPEC = PersonaSpec(
    mode=MODE,
    user_words=(80, 180),
    # Floor is at the 160-word average target, so no accepted pair can drag the
    # average below it. The prompt aims at the (160+250)/2 = 205 midpoint, which
    # leaves comfortable room above the target.
    eka_words=(160, 250),
    markers=(
        "strategy",
        "move",
        "timing",
        "alliance",
        "evidence",
        "power",
        "what is your next move",
    ),
    # The persona's own hard rule is to close on "what is the one move available
    # to you right now?", so a closing question is a legitimate marker in its
    # own right — and "move" above already covers the canonical phrasing.
    question_counts_as_marker=True,
    # Chanakya is a strategist; directive advice is the entire point of the
    # persona. Only reflection mode forbids it.
    forbid_advice=False,
    require_ends_with_question=False,
    require_verse=False,
    max_similarity=0.75,   # Jaccard; see DUPLICATE_THRESHOLD in _gen_quality.py
    min_marker_pass_rate=0.90,
    max_near_duplicates=0,  # binary guarantee: zero rows may duplicate an earlier row
    min_avg_eka_words=160,
)

EXTRA = (
    "The situation must contain a real power asymmetry — someone holds "
    "information, money, access, or leverage the user does not. "
    "Eka must name who holds what before proposing anything, and must never "
    "moralize about whether the situation is fair. "
)


if __name__ == "__main__":
    raise SystemExit(run(MODE, OUTPUT, REPORT, TOPIC_COUNTS, SPEC, EXTRA))
