"""Generate the Gita-mode training set: 600 quality-gated pairs, 10 topics.

    python ml/scripts/generate_gita_data.py              # generate / resume
    python ml/scripts/generate_gita_data.py --report-only # re-score, no API calls
    python ml/scripts/generate_gita_data.py --plan        # estimate, no API calls

Resume-safe, CPU only. See _gen_cli.py for the shared CLI and _gen_quality.py
for the five gates every pair must clear before it is written.

THIS PERSONA HAS THE STRICTEST GATE OF THE FOUR: require_verse=True means a
response with no chapter-and-verse citation is rejected outright, and the
quality report blocks training unless verse_reference_rate is exactly 1.0.
Expect a higher rejection and regeneration rate here than for the other three.

backend/prompts/gita.txt was raised from a 200- to a 250-word cap so the
150-250 range below does not train the model to violate its own system prompt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gen_cli import run
from _gen_quality import PersonaSpec

MODE = "gita"
OUTPUT = "gita_dataset.json"
REPORT = "gita_dataset_quality_report.json"

# Rebalanced to exactly 600. The previous version summed to 630 across these
# same ten topics, which quietly overshot the stated target by 5%.
TOPIC_COUNTS = {
    "fear of judgment": 60,
    "attachment to outcomes": 60,
    "finding dharma or purpose": 60,
    "dealing with grief or loss": 60,
    "inner conflict duty vs desire": 60,
    "spiritual doubt": 60,
    "detachment in relationships": 60,
    "accepting impermanence": 60,
    "self-knowledge": 60,
    "karma and consequences": 60,
}  # = 600

SPEC = PersonaSpec(
    mode=MODE,
    user_words=(80, 180),
    eka_words=(150, 250),
    # Markers match as `\b<marker>\w*\b`, so a marker only covers inflections it
    # is a genuine PREFIX of. "detachment" therefore missed "detached from the
    # outcome" and "detaching slowly" — both squarely on-persona — so the stem
    # "detach" is used instead, which covers all four forms. "attachment" is
    # listed separately because the persona's own prompt says "action without
    # attachment", which "detach" does not cover. Krishna and equanimity are
    # added because they are core persona vocabulary that was simply missing.
    markers=(
        # Stems, for the same prefix reason: "dharm"/"karm" also cover dharmic
        # and karmic, which "dharma"/"karma" do not.
        "dharm",
        "karm",
        "detach",
        "attachment",
        "lotus",
        "Arjuna",
        "Krishna",
        "equanimity",
    ),
    # A closing question does NOT satisfy the marker gate here. The persona is
    # defined by its vocabulary and its citations, not by its question shape, so
    # letting a question stand in would admit responses with no Gita content.
    question_counts_as_marker=False,
    # The hard requirement: every response cites a real chapter and verse.
    require_verse=True,
    # The persona explicitly forbids practical/business advice, but the advice
    # gate is tuned for reflection's therapeutic stance and would misfire on
    # spiritual instruction ("act without clinging to the fruit" reads as a
    # directive). Left off deliberately; the no-tactics rule is enforced by the
    # persona prompt rather than by regex.
    forbid_advice=False,
    require_ends_with_question=False,
    max_similarity=0.75,   # Jaccard; see DUPLICATE_THRESHOLD in _gen_quality.py
    min_marker_pass_rate=0.90,
    max_near_duplicates=0,  # binary guarantee: zero rows may duplicate an earlier row
    min_avg_eka_words=0,
)

EXTRA = (
    "Eka must cite at least one real Bhagavad Gita verse by chapter and verse "
    "number, woven naturally into the response — never as a block quote. "
    "Use the verse the situation actually calls for rather than defaulting to "
    "2.47 every time, and never give practical, tactical, or numerical advice. "
)


if __name__ == "__main__":
    raise SystemExit(run(MODE, OUTPUT, REPORT, TOPIC_COUNTS, SPEC, EXTRA))
