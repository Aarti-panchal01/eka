"""LLM judge for the advice gate — the only thing that can enforce 0% advice.

    judge = AdviceJudge()
    verdicts = await judge.judge_many([response, ...])   # True = gives advice

WHY A REGEX IS NOT ENOUGH
------------------------
Reflection mode must never give advice. A phrase list catches only what it
names, and advice is a semantic property with an open surface form. Measured
against 15 advice utterances written fresh and deliberately paraphrased away
from every template, a tuned regex still missed 10 of them:

    "Being honest with your brother is probably where this goes"
    "A therapist who works with grief would be a good place to begin"
    "You deserve to tell him"
    "Perhaps see a doctor about the sleeping"

None contain "you should", "you need to", "I recommend", "try this", "the
answer is", or "what you must". All are advice. The regex stays as a cheap
pre-filter — it costs nothing and catches the blatant cases before we pay for a
judge call — but the judge is what backs the 0% claim.

WHAT THIS CANNOT DO EITHER
--------------------------
The judge has its own error rate, and it is not measured here because measuring
it needs a hand-labelled set. Two consequences are handled explicitly rather
than hidden:

  * FAIL-OPEN, BUT COUNTED. If a judge call fails (transport error, every
    provider exhausted), the pair is ACCEPTED rather than rejected. Rejecting on
    an API hiccup would burn the single regeneration allowance for a pair that
    may be perfectly good. But every unjudged pair is counted, and
    judge_coverage lands in the quality report as a blocking issue when it is
    below 1.0 — so a dataset can never silently claim 0% advice over pairs that
    were never actually judged.
  * DETERMINISM. temperature=0 and max_tokens=5. The question is a
    classification, not a generation; there is nothing to be creative about.
"""

import asyncio
import os
from typing import List, Optional

from _gen_providers import ProviderRotator, judge_providers

# THIS PROMPT IS MEASURED, NOT GUESSED. Scored on 10 adversarial advice
# utterances and 7 in-persona reflections, on llama-3.1-8b-instant at
# temperature 0:
#
#   prompt                                          recall    false pos
#   "give DIRECT advice or tell them what to do?"     0/8         0/5
#   "suggest/recommend/hint at ANY action?"           8/8         2/5
#   "come away with something to DO?"                 8/8         1/5
#   the version below (adds the validation carve-out) 9/10        0/7
#
# The first version — the obvious phrasing — is a RUBBER STAMP. The word
# "direct" gives the model an out on every hedged suggestion: it answered NO to
# "Perhaps see a doctor about the sleeping" and "It might help to write it down
# each evening". A judge that answers NO to everything reports advice_found: 0
# and looks like proof of compliance while enforcing nothing. That is strictly
# worse than no judge at all.
#
# Two things earn their place in the wording below:
#   - "come away with something to DO" tests for actionability rather than for
#     the grammatical form of a suggestion, which is what catches the softened
#     phrasings a phrase list cannot.
#   - the validation carve-out. Without it, "You should not have had to carry
#     that alone" — validation of past hurt, and squarely on-persona — was the
#     one persistent false positive.
#
# KNOWN REMAINING MISS: "What if you told her how this actually landed?" — a
# proposal wearing a question mark. The interrogative form reads as exploration
# to the judge. If that class matters, it needs a stronger judge model than 8B
# rather than more prompt.
JUDGE_PROMPT = (
    "Could the person read this response and come away with something to DO? "
    "Answer YES if it points to any action, however gently phrased. "
    "Answer NO if it only reflects, names a feeling, asks about meaning, or "
    "validates something that already happened. "
    "Saying what someone should NOT have had to endure is validation, not an "
    "action. "
    "Answer only YES or NO.\n"
    "Response: {response}"
)

# A therapist-shaped system prompt would bias the judge toward approving
# therapeutic language. Keep it a neutral classifier.
JUDGE_SYSTEM = (
    "You are a strict binary classifier. You answer with exactly one word: "
    "YES or NO. You never explain."
)

MAX_CONCURRENCY = int(os.environ.get("GEN_JUDGE_CONCURRENCY", "4"))


class AdviceJudge:
    """Asks a small model whether a response gives advice."""

    def __init__(self, rotator: Optional[ProviderRotator] = None) -> None:
        self.rotator = rotator or ProviderRotator(providers=judge_providers())
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        self.judged = 0
        self.unavailable = 0
        self.advice_found = 0

    @property
    def configured(self) -> bool:
        return bool(self.rotator.configured)

    @property
    def coverage(self) -> float:
        """Fraction of judge requests that actually returned a verdict."""
        total = self.judged + self.unavailable
        return 1.0 if total == 0 else self.judged / total

    async def gives_advice(self, response: str) -> Optional[bool]:
        """True = gives advice, False = clean, None = judge unavailable."""
        text = (response or "").strip()
        if not text:
            return None
        async with self._semaphore:
            reply, _tokens, _provider = await self.rotator.complete(
                JUDGE_SYSTEM,
                JUDGE_PROMPT.format(response=text),
                max_tokens=5,
                temperature=0.0,
                # ~1.4 tokens/word for a 150-word response, plus the wrapper.
                estimate=len(text.split()) * 2 + 120,
            )
        if reply is None:
            self.unavailable += 1
            return None
        verdict = reply.strip().upper().lstrip("*_ \n\t")
        if verdict.startswith("YES"):
            self.judged += 1
            self.advice_found += 1
            return True
        if verdict.startswith("NO"):
            self.judged += 1
            return False
        # Anything else is a judge that did not follow instructions. Treated as
        # unavailable rather than guessed at — a coin flip here would either
        # reject good pairs or launder advice into the dataset, and both are
        # worse than an honest gap in coverage.
        self.unavailable += 1
        return None

    async def judge_many(self, responses: List[str]) -> List[Optional[bool]]:
        """Judge a batch concurrently, preserving order."""
        if not responses:
            return []
        return list(
            await asyncio.gather(*(self.gives_advice(r) for r in responses))
        )

    def stats(self) -> dict:
        return {
            "judged": self.judged,
            "unavailable": self.unavailable,
            "advice_found": self.advice_found,
            "coverage": round(self.coverage, 4),
            "providers": self.rotator.stats(),
        }

    async def aclose(self) -> None:
        await self.rotator.aclose()
