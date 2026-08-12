"""Quality gates for generated persona data, plus the quality report.

Every pair crosses five gates before it is allowed into a dataset:

    1. JSON shape      — user and eka_response are both non-empty strings
    2. Length          — per-persona word floors and ceilings
    3. Persona markers — the response actually sounds like the persona
    4. Structure       — persona-specific hard rules (verse ref, ends with a
                         question, contains no advice)
    5. Diversity       — not a near-duplicate of a pair we already have

A pair that fails any gate is rejected and queued for ONE regeneration attempt
with a prompt that names the specific deficiency. Fails twice, it is logged and
skipped — no infinite loops.

WHY GATE AT ALL: a LoRA copies its training data with uncomfortable fidelity.
An earlier unguarded pass produced responses averaging 88 words against a
100-180 word target and routinely ending in three stacked questions. Every one
of those defects would have become a permanent trait of the fine-tuned model.
Rejecting 10% of generations is much cheaper than discovering the drift after
a 3-hour Kaggle run.

A note on where these thresholds came from: they are product requirements, not
measurements. They are strict on purpose. If a persona's pass rate comes in
very low, the honest reading is usually that the prompt and the gate disagree
about what the persona is — fix the prompt, do not quietly loosen the gate.
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)

# --------------------------------------------------------------------------
# Bhagavad Gita citation detection.
#
# A plain two-number regex CANNOT do this job, and an earlier version of this
# file tried. `\b\d{1,2}[.:]\d{1,3}\b` produced 26 false positives in a 54-case
# regression table: "6.30 in the morning", "a 3:1 ratio", "2.5 years",
# "version 2.14", "Section 2.47 of the shareholders agreement", "18.78 lakh"
# all satisfied it. Since gita mode sets require_verse=True and the quality
# report blocks training unless verse_reference_rate == 1.0, that meant the
# strictest gate in the whole pipeline was effectively a no-op: any response
# containing any decimal passed as "cited".
#
# The core problem is that "2.5 years" is a syntactically valid chapter-2
# verse-5 reference. Syntax alone cannot separate the two, so three additional
# rules are required, and all three are load bearing:
#
#   1. RANGE VALIDATION  — chapter must be 1-18 and the verse must exist in
#      that chapter (chapter 12 has 20 verses, so "12.55" is not a citation).
#   2. UNIT GUARDS       — a number carrying a unit ("2.5 years", "6.30 a.m.",
#      "2.5x") or a label ("version", "Section", "Rs") is a quantity.
#   3. SOURCE CUE        — the sentence must actually name the source
#      (Krishna, Arjuna, Gita, chapter, verse, "tells", "says", ...).
#
# The replacement scores 0 false positives and 0 false negatives on the same
# 54 cases. Two residual false negatives were accepted deliberately, both
# telegraphic forms the persona prompt does not produce ("Krishna. 2.47. Act.");
# a false negative only costs one regeneration with a clear reason, while a
# false positive silently admits an uncited response into the training data and
# falsifies the 100% claim in the quality report.
# --------------------------------------------------------------------------

# Verse counts per chapter (Gita Press / BORI recension).
_BG_VERSE_COUNTS = {
    1: 47, 2: 72, 3: 43, 4: 42, 5: 29, 6: 47, 7: 30, 8: 28, 9: 34,
    10: 42, 11: 55, 12: 20, 13: 35, 14: 27, 15: 20, 16: 24, 17: 28, 18: 78,
}
_DASH = "\\-\u2010-\u2015\u2212"

# A numeral is a citation only if its sentence names the source.
_CITE_CUE_RE = re.compile(
    r"\b(?:bhagavad|gita|bg|krishna|arjuna|chapters?|verses?|shlokas?|slokas?|"
    r"tells|told|says|said|speaks|spoke|teaches|taught|warns|warned|reminds|"
    r"counsels|as\s+it\s+is\s+written|draw\s+on)\b",
    re.IGNORECASE,
)
# ...and only if it is not carrying a unit.
_QUANTITY_AFTER_RE = re.compile(
    r"\s*(?:x\b|%|percent|per\s*cent|years?\b|yrs?\b|months?\b|weeks?\b|"
    r"days?\b|hours?\b|hrs?\b|minutes?\b|mins?\b|seconds?\b|secs?\b|times\b|"
    r"lakhs?\b|crores?\b|million\b|billion\b|[km]\b|rupees\b|dollars\b|"
    r"met(?:re|er)s?\b|miles\b|kgs?\b|a\.?m\.?|p\.?m\.?|o'clock\b|ratio\b|"
    r"odds\b|probability\b|out\s+of\b)",
    re.IGNORECASE,
)
_QUANTITY_BEFORE_RE = re.compile(
    r"(?:version|ver\.?|\bv|section|clause|sec\.?|para|figure|fig\.?|table|"
    r"score|rs\.?|inr|usd|no\.?|#|\u20b9|\$)\s*$",
    re.IGNORECASE,
)
_NUMERIC_VERSE_RE = re.compile(
    r"(?<![\w.:])(?P<c>\d{1,2})\s*[.:]\s*(?P<v>\d{1,3})"
    r"(?:\s*[" + _DASH + r"]\s*(?P<v2>\d{1,3}))?"  # capture the FULL range
    r"(?!\s*[.:]\s*\d)"  # reject 1.2.3 / semver / 25.400.1
)
_WORDY_VERSE_RE = re.compile(
    r"\bchapter\s+(?P<c>\d{1,2}|[a-z]+)\s*"
    r"(?:[,;:" + _DASH + r"\u2014\u2013]|\s)\s*(?:and\s+)?"
    r"verses?\s+(?P<v>\d{1,3}|[a-z]+(?:[" + _DASH + r"][a-z]+)?)",
    re.IGNORECASE,
)

_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "twentieth": 20, "thirtieth": 30, "fortieth": 40,
    "fiftieth": 50, "sixtieth": 60, "seventieth": 70,
}


def _to_int(token: Optional[str]) -> Optional[int]:
    """Parse '47' or 'forty-seven'. Rejects '08' so dates do not slip through."""
    token = (token or "").strip().lower()
    if not token:
        return None
    if token.isdigit():
        return int(token) if token == str(int(token)) else None  # rejects "08"
    parts = re.split(r"[\s\u2010-\u2015\-]+", token)
    if len(parts) == 1:
        return _TENS.get(parts[0], _UNITS.get(parts[0]))
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _UNITS:
        return _TENS[parts[0]] + _UNITS[parts[1]]
    return None


def _verse_exists(chapter: int, verse: int) -> bool:
    return chapter in _BG_VERSE_COUNTS and 1 <= verse <= _BG_VERSE_COUNTS[chapter]


def _sentence_around(text: str, pos: int) -> str:
    """The sentence containing `pos` — the scope the source cue must appear in."""
    start = 0
    for match in re.finditer(r"[.!?\n]\s+", text):
        if match.end() <= pos:
            start = match.end()
        else:
            break
    tail = re.search(r"[.!?\n]\s", text[pos:])
    return text[start : pos + tail.start() if tail else len(text)]


def find_verse(text: str) -> Optional[str]:
    """Return the matched Bhagavad Gita citation, or None.

    Checks the spelled-out "Chapter 2, verse 47" form first (unambiguous, so it
    needs no cue), then numerals with the full guard stack.
    """
    text = text or ""
    for match in _WORDY_VERSE_RE.finditer(text):
        chapter, verse = _to_int(match.group("c")), _to_int(match.group("v"))
        if chapter is not None and verse is not None and _verse_exists(chapter, verse):
            return match.group(0)
    for match in _NUMERIC_VERSE_RE.finditer(text):
        chapter, verse = _to_int(match.group("c")), _to_int(match.group("v"))
        verse_end = _to_int(match.group("v2"))
        if chapter is None or verse is None or not _verse_exists(chapter, verse):
            continue
        if verse_end is not None and not _verse_exists(chapter, verse_end):
            continue
        if _QUANTITY_AFTER_RE.match(text, match.end()):
            continue
        if _QUANTITY_BEFORE_RE.search(text[max(0, match.start() - 12) : match.start()]):
            continue
        if _CITE_CUE_RE.search(_sentence_around(text, match.start())):
            return match.group(0)
    return None

# Advice-giving constructions that reflection mode must never use. Matched on
# the response only — a user is free to ask "should I quit?", and the whole
# point of the gate is that Eka must not answer it with a directive.
#
# Deliberately targets CONSTRUCTIONS, not the bare word "should": "I wonder
# what that should tell you" is a legitimate reflective move, while "you
# should" is not. Matching bare "should" rejected ~30% of otherwise-good
# reflection pairs in a spot check.
_ADVICE_RE = re.compile(
    r"\b(?:"
    r"you\s+should|you\s+need\s+to|you\s+ought\s+to|you\s+must|you\s+have\s+to|"
    r"you\s+shouldn't|you\s+can't|you\s+should\s+not|"
    r"my\s+advice|i\s+(?:would\s+)?(?:advise|suggest|recommend)|"
    r"here'?s\s+what\s+(?:you\s+should|to)\s+do|"
    r"(?:the\s+)?best\s+(?:thing|move)\s+(?:for\s+you\s+)?(?:would\s+be|is)|"
    r"try\s+to\s+|make\s+sure\s+(?:you|to)|"
    r"what\s+you\s+need\s+is"
    r")\b",
    re.IGNORECASE,
)


def words(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")


def word_count(text: str) -> int:
    return len(words(text))


def token_set(text: str) -> Set[str]:
    """Lowercased word set, used as the diversity proxy.

    Stopwords are kept, but NOT for the reason an earlier version of this
    docstring claimed. Measured over 300 pairs per class, removing them makes
    every class less similar, duplicates least of all:

        class                         with     without    delta
        clone (details re-rolled)     0.896     0.847     -0.049
        independent, same topic       0.476     0.363     -0.112
        different topic               0.311     0.165     -0.145

    So removal actually IMPROVES class separation (2.88x -> 5.13x). Both give
    AUC 1.000; keeping stopwords has a slightly larger absolute margin (+0.122
    vs +0.091), which is the only real reason to keep them. The cost is that
    stopword mass compresses every score into roughly [0.14, 0.97], and that
    compression is exactly what makes a high threshold brittle.
    """
    return {w.lower() for w in words(text)}


# ===========================================================================
# DUPLICATE THRESHOLD — 0.75 IS EMPIRICALLY DERIVED. DO NOT RAISE IT WITHOUT
# RE-MEASURING ON NEW DATA.
#
# Measured with JACCARD (|A n B| / |A u B|) on real generated data:
#
#   independent same-topic pairs (60 real pairs)  max 0.6923   p99 0.6047
#   clone, slot values re-rolled (300 pairs)      min 0.8039  mean 0.8348
#
# 0.75 sits in the gap between those two classes: 100% recall on clones, 0%
# false positives on independent pairs, with 0.058 of margin below the nearest
# real collision. At 0.85 clone recall collapses to 21%.
#
# CRITICAL IF YOU RETUNE: these are JACCARD numbers and they are not
# interchangeable with cosine. Jaccard is always the lower of the two — for
# equal-sized sets, cosine 0.75 corresponds to Jaccard 0.60. An earlier round of
# measurement produced 0.75 from COSINE data (independent max 0.728), and
# carrying that number across to Jaccard unexamined would have been a coincidence
# rather than a calibration. Measure the metric you are actually using.
# ===========================================================================
DUPLICATE_THRESHOLD = 0.75
# How many recent pairs to compare against during generation. The full sweep is
# affordable (~6.7s at n=1000) and is what the final report uses; this window
# only bounds the per-pair cost on the hot path.
DUPLICATE_WINDOW = 200


def jaccard_overlap(a: Set[str], b: Set[str]) -> float:
    """|A n B| / |A u B|. The duplicate metric of record for this module."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def check_duplicates(
    new_pair: dict,
    existing_pairs: Sequence[dict],
    threshold: float = DUPLICATE_THRESHOLD,
    window: Optional[int] = DUPLICATE_WINDOW,
) -> Optional[float]:
    """Return the offending similarity if new_pair duplicates an existing one.

    Returns None when the pair is sufficiently novel, so the caller can report
    the actual score rather than a bare boolean.

    Tokenised with words() rather than str.split() so that "runway." and
    "runway" are the same token — split() leaves punctuation attached, which
    silently lowers similarity and lets duplicates through. Measured difference
    on clean template data is nil (0.8348 either way); it matters on real LLM
    output, which is full of commas and em dashes.

    window=None compares against every existing pair. That is what backs the
    unique_pair_guarantee field in the quality report: a sliding window cannot
    guarantee uniqueness over a set larger than the window.
    """
    new_words = token_set(new_pair.get("user", ""))
    if not new_words:
        return None
    candidates = existing_pairs if window is None else existing_pairs[-window:]
    for existing in candidates:
        score = jaccard_overlap(new_words, token_set(existing.get("user", "")))
        if score > threshold:
            return score
    return None


def cosine_overlap(a: Set[str], b: Set[str]) -> float:
    """Cosine similarity of two binary word-presence vectors.

    |A n B| / sqrt(|A| * |B|). This is the "simple word overlap as proxy" the
    spec asks for. It is a genuine cosine — on binary vectors the dot product
    is the intersection size and each norm is sqrt(set size) — it is just
    computed on presence rather than TF-IDF weights, so it over-reports
    similarity for messages that share common words. That bias is safe here:
    it rejects a few acceptable pairs rather than admitting duplicates.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


@dataclass
class PersonaSpec:
    """Everything mode-specific that the gates need."""

    mode: str
    user_words: Tuple[int, int]
    eka_words: Tuple[int, int]
    # Response must contain at least one of these (case-insensitive substring,
    # matched on word boundaries where the marker is a single word).
    markers: Sequence[str] = ()
    # Counts as satisfying `markers` if the response ends in a question. Used
    # by founder, whose spec lists "specific question at end" as a marker.
    question_counts_as_marker: bool = False
    require_verse: bool = False
    require_ends_with_question: bool = False
    forbid_advice: bool = False
    # An explicit phrase blacklist. When non-empty this REPLACES the broad
    # _ADVICE_RE heuristic for this persona — a named, auditable list is easier
    # to reason about than a regex, and it is what the reflection spec calls
    # for. The tradeoff is that it only catches what it lists: softened advice
    # ("have you considered leaving", "it might help to") passes a phrase list
    # and would be caught by _ADVICE_RE. Set forbid_advice=True as well to run
    # both, at the cost of a higher rejection rate.
    forbidden_phrases: Sequence[str] = ()
    # Persona-specific preamble for the regeneration prompt. Overrides the
    # generic "a previous attempt was rejected" framing when set.
    regen_hint: str = ""
    # Run an LLM judge over every response that clears the cheap gates, asking
    # whether it gives advice. This is the only thing that can actually enforce
    # a 0%-advice requirement: advice is a semantic property with an open
    # surface form, and a phrase list catches only what it names (measured: a
    # tuned regex missed 10 of 15 freshly paraphrased advice utterances).
    # Costs one extra API call per surviving pair — see AdviceJudge.
    llm_judge_advice: bool = False
    max_similarity: float = 0.85
    # Report thresholds. Below these, the caller should stop and ask a human
    # rather than train on the result.
    min_marker_pass_rate: float = 0.90
    # How many rows may duplicate an earlier row (nearest-neighbour similarity
    # above max_similarity). Zero is a meaningful, reachable target, which is
    # why this replaced the old min_diversity_score mean — see the long note in
    # quality_report() for why a mean cannot be gated on.
    max_near_duplicates: int = 0
    min_avg_eka_words: int = 0

    def marker_hit(self, response: str) -> Optional[str]:
        """Return the marker that matched, or None."""
        low = (response or "").lower()
        for marker in self.markers:
            m = marker.lower()
            # Multi-word markers are phrases; single words get boundaries so
            # "burn" does not match "burnout" and inflate the pass rate.
            if " " in m or "-" in m:
                if m in low:
                    return marker
            elif re.search(rf"\b{re.escape(m)}\w*\b", low):
                return marker
        if self.question_counts_as_marker and ends_with_question(response):
            return "question-at-end"
        return None


# An obligation phrase inside a wh-question is not advice — it is the core
# Socratic move this persona is built on. reflection.txt itself prescribes
# "What would you have to give up to believe that?" as a canonical question, and
# a naive substring check flags that as advice. Measured on a 40-response
# in-persona corpus, ~20% of GOOD reflection responses were rejected this way,
# and two tripped inside the mandated closing question itself. Left unfixed, the
# regeneration prompt then tells the model it "gave advice" when it did not,
# which trains the generator away from its best questions — a data-quality
# regression, not just wasted throughput.
_WH_CLAUSE_RE = re.compile(
    r"^\s*(?:and|but|so|yet)?\s*"
    r"(?:what|how|why|who|whose|whom|where|when)\b",
    re.IGNORECASE,
)
# "You should not have had to carry that alone" is validation of past hurt, not
# a directive about future action.
_COUNTERFACTUAL_RE = re.compile(
    r"\bshould\s*n(?:o|')?t?\s+have\b|\bshould\s+not\s+have\b", re.IGNORECASE
)


# "You really should talk to her" defeats a plain "you should" substring, and
# adverb insertion is the single easiest way for a directive to slip past a
# phrase list. Collapsing the adverb restores the match without lengthening the
# list combinatorially.
_ADVERB_INSERT_RE = re.compile(
    r"\byou\s+(?:really|just|probably|actually|definitely|honestly|simply|truly|"
    r"certainly|maybe|perhaps|also|still|now|only|literally|genuinely)\s+"
    r"(should|need|must|have|ought|can|might|could|would)\b",
    re.IGNORECASE,
)


def _normalize_quotes(text: str) -> str:
    """Fold typographic apostrophes and collapse inserted adverbs.

    LLM output is full of U+2019, and a pattern written with a straight quote
    silently fails to match it — which disabled several of the strictest
    alternatives in _ADVICE_RE on real generations.
    """
    text = (text or "").replace("’", "'").replace("‘", "'")
    return _ADVERB_INSERT_RE.sub(r"you \1", text)


def _sentences(text: str) -> List[Tuple[int, str]]:
    """(offset, sentence) pairs, so a hit can be located in its own clause."""
    out: List[Tuple[int, str]] = []
    start = 0
    for match in re.finditer(r"[.!?]+[\s\"')\]]*", text):
        out.append((start, text[start : match.end()]))
        start = match.end()
    if start < len(text):
        out.append((start, text[start:]))
    return out


def advice_hit(spec: "PersonaSpec", response: str) -> Optional[str]:
    """Return the offending phrase if the response gives advice, else None.

    A persona's explicit `forbidden_phrases` list takes precedence over the
    broad `_ADVICE_RE` heuristic so that a named list stays authoritative and
    auditable. Both can be active at once if forbid_advice is also set.

    IMPORTANT — WHAT THIS CANNOT DO: advice is a semantic property with an open
    surface form, and no phrase list closes it. Measured against 15 freshly
    written advice utterances deliberately paraphrased away from every template,
    a tuned version of this gate still missed 10 ("Being honest with your
    brother is probably where this goes", "A therapist who works with grief
    would be a good place to begin"). Treat a 0.0 hit rate as "no listed phrase
    appeared", NOT as proof the data contains no advice — which is why the
    report field is named advice_regex_hit_rate rather than
    contains_advice_rate. An LLM judge over the finished dataset is the only
    thing that can actually enforce the 0% requirement.
    """
    text = _normalize_quotes(response)
    if spec.forbidden_phrases:
        lowered = text.lower()
        for phrase in spec.forbidden_phrases:
            needle = phrase.lower()
            position = lowered.find(needle)
            while position != -1:
                # Locate the sentence this hit sits in and exempt the two
                # constructions that are reflective rather than directive.
                clause = ""
                for offset, sentence in _sentences(text):
                    if offset <= position < offset + len(sentence):
                        clause = sentence
                        break
                # The wh-exemption requires an actual QUESTION. A wh-cleft
                # declarative — "And what you must do is begin." — is a
                # directive wearing an interrogative word, so the trailing
                # question mark is what separates the Socratic move from the
                # instruction.
                stripped = clause.strip()
                is_wh_question = bool(_WH_CLAUSE_RE.match(clause)) and stripped.endswith(
                    "?"
                )
                exempt = is_wh_question or bool(_COUNTERFACTUAL_RE.search(clause))
                if not exempt:
                    return phrase
                position = lowered.find(needle, position + 1)
    if spec.forbid_advice:
        found = _ADVICE_RE.search(text)
        if found:
            return found.group(0).strip()
    return None


def ends_with_question(text: str) -> bool:
    stripped = (text or "").strip().rstrip('"\'”’)')
    return stripped.endswith("?")


def count_trailing_questions(text: str) -> int:
    """How many question sentences the response ends with.

    The personas are specified to end with EXACTLY one question. Stacking two
    or three is the single most common drift in generated persona data, so it
    is measured rather than assumed.
    """
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    trailing = 0
    for sentence in reversed(sentences):
        if sentence.strip().endswith("?"):
            trailing += 1
        elif sentence.strip():
            break
    return trailing


class DiversityIndex:
    """Incremental near-duplicate detector over user messages.

    A single flat sweep, deliberately. Two "optimisations" were removed after
    measurement showed both were harmful:

    1. A LENGTH PREFILTER that skipped pairs whose set sizes were too far apart
       for cosine to reach the threshold (sound bound: |A n B| <= min(|A|,|B|),
       so sim <= sqrt(min/max)). The bound is correct and tight, but the length
       gate constrains user messages to 100-200 words, so token sets land in
       ~93..123 and the smallest possible ratio is 0.756 — above the 0.7225
       needed to skip at a 0.85 threshold. Measured skip rate at n=200/500/1000:
       0.0%. It was pure overhead (+7-19% wall clock) AND a latent correctness
       trap: its soundness depends on this index's max_similarity matching the
       spec's, and nothing enforced that, so lowering a spec threshold would
       have silently started missing duplicates.

    2. A PER-TOPIC FIRST PASS, on the theory that duplicates share a topic and
       the global sweep could then stop early. It could not — the global list
       contains the same-topic entries, so it re-scanned them. Every same-topic
       message was compared twice for no benefit (1.06x candidate visits, and
       de-duplicating changed the returned score on 0/200 rows).

    Cost is O(n^2) and that is fine here: a full 1,000-pair build takes ~6.7s of
    set intersections against the ~1.5-2h of API calls that produce the data.
    A sliding `window` bounds the per-pair cost on the generation hot path; the
    quality report passes window=None because a windowed check cannot guarantee
    uniqueness over a set larger than the window.

    Similarity is JACCARD, not cosine — see the DUPLICATE_THRESHOLD note above
    for the calibration and for why the two are not interchangeable.
    """

    def __init__(
        self,
        max_similarity: float = DUPLICATE_THRESHOLD,
        window: Optional[int] = DUPLICATE_WINDOW,
    ) -> None:
        self.max_similarity = max_similarity
        self.window = window
        self._all: List[Set[str]] = []

    def add(self, user_message: str, topic: str = "") -> None:
        # `topic` is accepted and ignored — see note 2 above. Kept in the
        # signature because callers pass it and it documents intent at the call
        # site.
        tokens = token_set(user_message)
        if not tokens:
            return
        self._all.append(tokens)

    def worst_similarity(
        self, user_message: str, topic: str = "", early_exit: bool = True
    ) -> float:
        """Highest Jaccard similarity to anything already indexed.

        early_exit=True stops as soon as the threshold is exceeded, which is all
        the gate needs. Pass early_exit=False when the TRUE maximum matters —
        the quality report does, because a truncated maximum under-reports
        similarity and therefore flatters the dataset.
        """
        tokens = token_set(user_message)
        if not tokens:
            return 0.0
        pool = self._all if self.window is None else self._all[-self.window :]
        worst = 0.0
        for other in pool:
            score = jaccard_overlap(tokens, other)
            if score > worst:
                worst = score
                if early_exit and worst > self.max_similarity:
                    return worst
        return worst

    def __len__(self) -> int:
        return len(self._all)


def validate_pair(
    pair: dict,
    spec: PersonaSpec,
    diversity: Optional[DiversityIndex] = None,
    topic: str = "",
) -> List[str]:
    """Return a list of human-readable failure reasons. Empty list = accepted.

    Reasons are phrased so they can be pasted straight into a regeneration
    prompt — that is what makes the "name exactly what was missing" retry work.
    """
    reasons: List[str] = []

    # --- gate 1: JSON shape ------------------------------------------------
    user = pair.get("user")
    eka = pair.get("eka_response")
    if not isinstance(user, str) or not user.strip():
        return ["'user' is missing or empty"]
    if not isinstance(eka, str) or not eka.strip():
        return ["'eka_response' is missing or empty"]

    # --- gate 2: length ----------------------------------------------------
    u_count, e_count = word_count(user), word_count(eka)
    u_lo, u_hi = spec.user_words
    e_lo, e_hi = spec.eka_words
    if u_count < u_lo:
        reasons.append(
            f"the user message is {u_count} words, below the {u_lo}-word minimum"
        )
    elif u_count > u_hi:
        reasons.append(
            f"the user message is {u_count} words, above the {u_hi}-word maximum"
        )
    if e_count < e_lo:
        reasons.append(
            f"the response is {e_count} words, below the {e_lo}-word minimum"
        )
    elif e_count > e_hi:
        reasons.append(
            f"the response is {e_count} words, above the {e_hi}-word maximum"
        )

    # --- gate 3: persona markers ------------------------------------------
    if spec.markers or spec.question_counts_as_marker:
        if spec.marker_hit(eka) is None:
            shown = ", ".join(spec.markers[:8])
            reasons.append(
                f"the response contains none of the required {spec.mode} markers "
                f"({shown})"
            )

    # --- gate 4: structural rules -----------------------------------------
    if spec.require_verse and find_verse(eka) is None:
        reasons.append(
            "the response has no Bhagavad Gita chapter-and-verse reference "
            "(e.g. '2.47')"
        )
    if spec.require_ends_with_question and not ends_with_question(eka):
        reasons.append("the response does not end with a question")
    offending = advice_hit(spec, eka)
    if offending:
        reasons.append(
            f"the response gives direct advice ('{offending}') when it must "
            f"only reflect and ask questions"
        )
    # Applies to every persona: the prompts all specify exactly one closing
    # question, and stacked questions are the most common drift.
    trailing = count_trailing_questions(eka)
    if trailing > 1:
        reasons.append(
            f"the response ends with {trailing} stacked questions; it must end "
            f"with exactly one"
        )

    # --- gate 5: diversity -------------------------------------------------
    if diversity is not None:
        score = diversity.worst_similarity(user, topic)
        if score > spec.max_similarity:
            reasons.append(
                f"the user message is {score:.2f} cosine-similar to an existing "
                f"pair (limit {spec.max_similarity:.2f}); it needs a different "
                f"person, city, and numbers"
            )

    return reasons


def topic_of(row: dict) -> str:
    tags = row.get("tags") or []
    return tags[0] if tags else row.get("topic", "")


def revalidate_existing(
    rows: List[dict], spec: PersonaSpec
) -> Tuple[List[dict], List[dict], DiversityIndex]:
    """Split rows already on disk into (kept, dropped) and build the index.

    Needed because the quality floors can change between runs. Pairs generated
    under a looser spec must not be silently inherited — they would count
    toward the total and drag the reported averages below the targets while
    looking like progress.
    """
    diversity = DiversityIndex(spec.max_similarity)
    kept: List[dict] = []
    dropped: List[dict] = []
    for row in rows:
        topic = topic_of(row)
        # Check against already-kept rows only, so one bad row does not cascade.
        reasons = validate_pair(row, spec, diversity, topic)
        if reasons:
            row = dict(row)
            row["_reasons"] = reasons
            dropped.append(row)
            continue
        kept.append(row)
        diversity.add(row["user"], topic)
    return kept, dropped, diversity


def quality_report(
    rows: List[dict],
    spec: PersonaSpec,
    topic_counts: Dict[str, int],
    generated: int = 0,
    rejected: int = 0,
    regen_attempted: int = 0,
    regen_succeeded: int = 0,
) -> dict:
    """Build the report that decides whether this dataset is trainable."""
    total = len(rows)
    if total == 0:
        return {"total_pairs": 0, "verdict": "empty", "blocking_issues": ["no pairs"]}

    u_lengths = [word_count(r.get("user", "")) for r in rows]
    e_lengths = [word_count(r.get("eka_response", "")) for r in rows]

    # Only meaningful when the persona actually HAS a marker gate. Without this
    # guard a spec with no markers and question_counts_as_marker=False reports
    # persona_marker_pass_rate = 0.0 and blocks training over a gate it never
    # opted into. Gate 3 in validate_pair already guards this; the report did not.
    has_marker_gate = bool(spec.markers) or spec.question_counts_as_marker
    marker_hits = (
        sum(1 for r in rows if spec.marker_hit(r.get("eka_response", "")) is not None)
        if has_marker_gate
        else total
    )
    verse_hits = sum(
        1 for r in rows if find_verse(r.get("eka_response", "")) is not None
    )
    question_ends = sum(1 for r in rows if ends_with_question(r.get("eka_response", "")))
    advice_hits = sum(
        1 for r in rows if advice_hit(spec, r.get("eka_response", "")) is not None
    )

    # NEAR-DUPLICATE COUNT, not a mean diversity score.
    #
    # The old metric was 1 - mean(nearest-neighbour similarity), gated at 0.85.
    # It was dropped because a mean over a MAX falls monotonically as the corpus
    # grows — every message added gives every other message another chance at a
    # closer neighbour. Measured: n=25 -> 0.55, n=200 -> 0.43, n=1000 -> 0.35,
    # and the real 50-row founder set scored 0.4369. A 0.85 floor demanded a mean
    # nearest-neighbour similarity of 0.15 and could not pass on any real
    # dataset, so the verdict read needs_review every run — which teaches the
    # operator to ignore it.
    #
    # A count is a binary guarantee instead: zero rows duplicating an earlier row
    # is both meaningful and reachable, and it is what unique_pair_guarantee
    # reports. window=None so this is a FULL sweep — the generation hot path uses
    # a 200-pair window for speed, but a windowed check cannot guarantee
    # uniqueness over 1,000 rows, and this field claims a guarantee.
    index = DiversityIndex(spec.max_similarity, window=None)
    near_duplicate_count = 0
    worst_seen = 0.0
    for row in rows:
        topic = topic_of(row)
        if len(index):
            # early_exit=False: the gate's truncated maximum under-reports
            # similarity, which flatters the dataset.
            score = index.worst_similarity(row.get("user", ""), topic, early_exit=False)
            worst_seen = max(worst_seen, score)
            if score > spec.max_similarity:
                near_duplicate_count += 1
        index.add(row.get("user", ""), topic)

    distribution: Dict[str, int] = {}
    for row in rows:
        topic = topic_of(row)
        distribution[topic] = distribution.get(topic, 0) + 1

    over_quota = {
        topic: {"have": distribution.get(topic, 0), "quota": quota}
        for topic, quota in topic_counts.items()
        if distribution.get(topic, 0) > quota * 1.10
    }
    missing_topics = [t for t in topic_counts if t not in distribution]

    attempted = generated + rejected
    report = {
        "total_pairs": total,
        "target_pairs": sum(topic_counts.values()),
        "avg_user_length": round(sum(u_lengths) / total, 1),
        "avg_eka_length": round(sum(e_lengths) / total, 1),
        "min_user_length": min(u_lengths),
        "min_eka_length": min(e_lengths),
        "persona_marker_pass_rate": round(marker_hits / total, 4),
        # Binary uniqueness guarantee, replacing the old fuzzy diversity_score.
        "near_duplicate_count": near_duplicate_count,
        "near_duplicate_rate": round(near_duplicate_count / total, 4),
        "unique_pair_guarantee": near_duplicate_count == 0,
        "max_pair_similarity": round(worst_seen, 4),
        "duplicate_threshold": spec.max_similarity,
        "topic_distribution": dict(sorted(distribution.items())),
        "topics_over_quota_10pct": over_quota,
        "topics_missing": missing_topics,
        "rejection_rate": round(rejected / attempted, 4) if attempted else 0.0,
        "regeneration_success_rate": (
            round(regen_succeeded / regen_attempted, 4) if regen_attempted else 0.0
        ),
        "ends_with_question_rate": round(question_ends / total, 4),
        "advice_regex_hit_rate": round(advice_hits / total, 4),
        "verse_reference_rate": round(verse_hits / total, 4),
        "counts": {
            "generated_accepted": generated,
            "rejected": rejected,
            "regen_attempted": regen_attempted,
            "regen_succeeded": regen_succeeded,
        },
        "thresholds": {
            "min_marker_pass_rate": spec.min_marker_pass_rate,
            "max_near_duplicates": spec.max_near_duplicates,
            "duplicate_threshold": spec.max_similarity,
            "min_avg_eka_words": spec.min_avg_eka_words,
        },
    }

    # --- verdict ---------------------------------------------------------
    blocking: List[str] = []
    if report["persona_marker_pass_rate"] < spec.min_marker_pass_rate:
        blocking.append(
            f"persona_marker_pass_rate {report['persona_marker_pass_rate']:.3f} "
            f"< {spec.min_marker_pass_rate}"
        )
    if near_duplicate_count > spec.max_near_duplicates:
        blocking.append(
            f"near_duplicate_count {near_duplicate_count} > "
            f"{spec.max_near_duplicates} (max pair similarity "
            f"{report['max_pair_similarity']:.3f} vs threshold "
            f"{spec.max_similarity}) — unique_pair_guarantee is false"
        )
    if spec.min_avg_eka_words and report["avg_eka_length"] < spec.min_avg_eka_words:
        blocking.append(
            f"avg_eka_length {report['avg_eka_length']} < {spec.min_avg_eka_words}"
        )
    if spec.require_verse and report["verse_reference_rate"] < 1.0:
        blocking.append(
            f"verse_reference_rate {report['verse_reference_rate']:.3f} < 1.0"
        )
    if spec.require_ends_with_question and report["ends_with_question_rate"] < 1.0:
        blocking.append(
            f"ends_with_question_rate {report['ends_with_question_rate']:.3f} < 1.0"
        )
    if (spec.forbid_advice or spec.forbidden_phrases) and report[
        "advice_regex_hit_rate"
    ] > 0.0:
        blocking.append(
            f"advice_regex_hit_rate {report['advice_regex_hit_rate']:.3f} > 0.0"
        )
    if over_quota:
        blocking.append(f"{len(over_quota)} topic(s) exceed quota by >10%")
    if total < sum(topic_counts.values()):
        blocking.append(f"{sum(topic_counts.values()) - total} pairs short of target")

    report["blocking_issues"] = blocking
    report["verdict"] = "ok_to_train" if not blocking else "needs_review"
    return report


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def print_report(report: dict) -> None:
    print("\n" + "=" * 66)
    print("  QUALITY REPORT")
    print("=" * 66)
    for key in (
        "total_pairs",
        "target_pairs",
        "avg_user_length",
        "avg_eka_length",
        "persona_marker_pass_rate",
        "near_duplicate_count",
        "near_duplicate_rate",
        "unique_pair_guarantee",
        "max_pair_similarity",
        "rejection_rate",
        "regeneration_success_rate",
        "ends_with_question_rate",
        "advice_regex_hit_rate",
        "verse_reference_rate",
    ):
        if key in report:
            print(f"  {key:<28} {report[key]}")
    if report.get("topics_over_quota_10pct"):
        print(f"  topics over quota            {report['topics_over_quota_10pct']}")
    if report.get("topics_missing"):
        print(f"  topics missing               {report['topics_missing']}")
    print("-" * 66)
    if report.get("verdict") == "ok_to_train":
        print("  VERDICT: ok_to_train")
    else:
        print("  VERDICT: needs_review — do NOT train until these are resolved:")
        for issue in report.get("blocking_issues", []):
            print(f"    - {issue}")
    print("=" * 66 + "\n")
