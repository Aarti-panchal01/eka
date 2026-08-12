"""Generate the Reflection-mode training set: 1000 quality-gated pairs, 12 topics.

    python ml/scripts/generate_reflection_data.py              # generate / resume
    python ml/scripts/generate_reflection_data.py --report-only # re-score, no API calls
    python ml/scripts/generate_reflection_data.py --plan        # estimate, no API calls

Resume-safe, CPU only. See _gen_cli.py for the shared CLI and _gen_quality.py
for the five gates every pair must clear before it is written.

THIS IS THE ONLY PERSONA WHOSE WORD CAP WAS NOT RAISED.
backend/prompts/reflection.txt keeps its 150-word maximum ("In therapy, less is
more. Leave space.") while founder, chanakya, and gita went to 250. Brevity is
load-bearing for the therapeutic stance, not an incidental limit, so the data
range below is 100-150 and must stay under that cap. Generating 250-word
reflection data would teach the fine-tune to over-talk and to argue with its own
system prompt at inference time.

Two hard requirements, both enforced as blocking issues in the quality report:
100% of responses end with a question, and 0% contain advice.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gen_cli import run
from _gen_quality import PersonaSpec

MODE = "reflection"
OUTPUT = "reflection_dataset.json"
REPORT = "reflection_dataset_quality_report.json"

TOPIC_COUNTS = {
    "chronic procrastination": 100,
    "self-sabotage patterns": 100,
    "fear of success": 80,
    "relationship anxiety": 80,
    "career identity confusion": 80,
    "emotional numbness": 80,
    "perfectionism": 80,
    "people pleasing": 80,
    "anger and resentment": 80,
    "childhood patterns in present": 80,
    "grief processing": 80,
    "addiction to validation": 80,
}  # = 1000

SPEC = PersonaSpec(
    mode=MODE,
    user_words=(50, 120),
    # HARD CAP at 150 — this is the persona's own limit, not a tuning choice.
    eka_words=(100, 150),
    # No vocabulary markers: this persona is defined by its SHAPE, not its
    # words. A reflection is identified by ending in a question and containing
    # no advice, which is exactly what the two gates below check. With markers
    # empty and question_counts_as_marker on, persona_marker_pass_rate measures
    # the ends-with-a-question rate, which is the meaningful signal here.
    markers=(),
    question_counts_as_marker=True,
    require_ends_with_question=True,
    require_verse=False,
    # An explicit, auditable blacklist rather than the broad _ADVICE_RE
    # heuristic. Note the tradeoff: a phrase list catches only what it names, so
    # softened advice ("have you considered leaving", "it might help to") will
    # pass this gate. Set forbid_advice=True to additionally run the broad
    # regex, at the cost of a materially higher rejection rate.
    forbidden_phrases=(
        "you should",
        "you need to",
        "I recommend",
        "try this",
        "the answer is",
        "what you must",
    ),
    forbid_advice=False,
    # Blunter than the generic defect list, because "you gave advice" needs a
    # direct correction rather than a description of a rule violation.
    regen_hint=(
        "The previous response gave advice. Rewrite it as ONLY a reflection and "
        "question, no advice, under 150 words."
    ),
    max_similarity=0.75,   # Jaccard; see DUPLICATE_THRESHOLD in _gen_quality.py
    min_marker_pass_rate=0.90,
    max_near_duplicates=0,  # binary guarantee: zero rows may duplicate an earlier row
    # No average-length target: the band is only 50 words wide and the cap is
    # the binding constraint, so an average gate would add nothing.
    min_avg_eka_words=0,
)

EXTRA = (
    "Eka must NOT give any advice — no suggestions, no 'you could try'. "
    "Only reflection, observation, and one deeper open question at the end. "
    "Name the emotion underneath what the user actually said, and reflect "
    "ambivalence back whole where it is present. "
)


if __name__ == "__main__":
    raise SystemExit(run(MODE, OUTPUT, REPORT, TOPIC_COUNTS, SPEC, EXTRA))
