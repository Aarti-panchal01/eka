"""Generate 2000 labelled examples for the complexity router. NO API NEEDED.

    python ml/scripts/generate_complexity_data.py

The complexity classifier decides how much context Eka pulls for a message:

    simple  -> 1 memory,  2 history,  pool 10
    normal  -> 2 memories, 3 history, pool 15
    complex -> 3 memories, 5 history, pool 25
    deep    -> 5 memories, 7 history, pool 40

Templates rather than an LLM, deliberately: the label is a function of
structure and length, so templates give perfectly clean labels for free.
"""

import json
import random
import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode ✓/✅/⏳.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPTS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPTS_DIR.parent / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = DATASETS_DIR / "complexity_labeled.jsonl"

random.seed(17)
PER_CLASS = 500

# ------------------------------------------------------------------ SIMPLE
GREETINGS = [
    "hi", "hello", "hey eka", "hey", "good morning", "good evening",
    "what's up", "yo", "namaste", "morning", "hi eka", "you there",
    "hey there", "sup", "hello again", "back again", "good night",
    "hi there", "evening", "afternoon", "hey again", "still awake",
    "long time", "i'm back", "hello eka", "greetings", "hola",
]
ONE_WORDERS = [
    "help", "advice", "guidance", "thoughts", "ideas", "suggestions",
    "okay", "continue", "go on", "more", "why", "how", "explain",
    "thanks", "got it", "makes sense", "hmm", "interesting", "true",
    "right", "sure", "and", "so", "then", "wait", "really", "elaborate",
    "example", "again", "next", "fair", "noted", "understood", "wow",
]
DEFINITIONS = [
    "what is dharma", "who was Chanakya", "what is PMF", "define runway",
    "what is karma", "what is burn rate", "what is CAC", "what is LTV",
    "what is moksha", "define pivot", "what is a term sheet", "what is ARR",
    "what is nishkama karma", "define bootstrapping", "what is churn",
    "who is Krishna", "what is the Arthashastra", "define equity",
    "what is a cap table", "what is CBT", "what is ACT therapy",
    "what is a seed round", "define unit economics", "what is MRR",
    "what is a moat", "define TAM", "what is gross margin",
    "who was Arjuna", "what is the Gita", "define vesting",
    "what is a SAFE note", "what is dilution", "define traction",
    "what is a KPI", "what is stoicism", "define detachment",
    "what is a lean canvas", "who is Arthashastra by", "what is JTBD",
    "define first principles", "what is default alive",
    "what is a bridge round", "define product led growth",
    "what is samsara", "define atman", "what is a north star metric",
    "what is motivational interviewing", "define burnout",
]
SIMPLE_SUFFIX = [
    "", "?", " ?", " eka", " please", " quickly", " in short", "!",
    " briefly", " one line",
]

# ------------------------------------------------------------------ NORMAL
EMOTIONS = [
    "anxious", "excited", "confused", "frustrated", "stuck", "worried",
    "hopeful", "overwhelmed", "uncertain", "angry", "tired", "restless",
    "nervous", "conflicted", "disappointed", "hesitant", "drained",
    "impatient", "doubtful", "torn",
]
TOPICS = [
    "my startup idea", "the funding round", "my co-founder", "our pricing",
    "hiring my first engineer", "the product launch", "my resignation",
    "our competitor", "the demo day", "our churn rate", "the pivot",
    "my burn rate", "the investor call", "our first customer",
    "the roadmap", "my equity split", "the board meeting", "our runway",
    "the acquisition offer", "my team's morale", "the tech stack choice",
    "our go-to-market plan", "the beta feedback", "my personal savings",
    "the next milestone",
]
RELATIONSHIPS = [
    "manager", "co-founder", "partner", "father", "mother", "best friend",
    "teammate", "mentor", "roommate", "sibling", "client", "investor",
]
BEHAVIORS = [
    "dismissing my ideas", "taking credit for my work", "avoiding my calls",
    "changing the plan without telling me", "overpromising to clients",
    "comparing me to others", "going quiet for days",
    "questioning every decision I make", "making commitments I have to keep",
    "shutting down every disagreement",
]
ACTIONS = [
    "say no", "ask for a raise", "push back", "set a boundary",
    "have the hard conversation", "walk away", "renegotiate",
    "delegate this", "admit I was wrong", "make the call",
]
SITUATIONS_SHORT = [
    "everyone is watching", "there's money on the line",
    "I'm the youngest in the room", "we're already behind schedule",
    "I don't have the full picture", "they hold all the leverage",
    "I've already committed publicly", "the team is looking to me",
]

# ----------------------------------------------------------------- COMPLEX
LONG_SITUATIONS = [
    "working two jobs", "running the company alone", "carrying the whole team",
    "living off my savings", "commuting three hours a day",
    "taking care of my parents while building this",
    "shipping features nobody uses", "chasing the same client",
    "rebuilding the product from scratch", "covering for a co-founder",
]
TIMEFRAMES = [
    "eight months", "almost two years", "the last year and a half",
    "six months", "three years", "since college", "since the pandemic",
]
CONSEQUENCES = [
    "my health has started slipping", "I barely speak to my friends anymore",
    "I've stopped enjoying the work", "my savings are nearly gone",
    "I snap at people I love", "I can't sleep before Mondays",
    "I've missed every deadline I set for myself",
    "I don't recognise the person I've become",
]
BALANCE_A = [
    "growth", "speed", "the company", "ambition", "what I owe my team",
    "what my parents expect", "the money", "the vision",
]
BALANCE_B = [
    "my health", "my relationship", "staying honest with myself",
    "the life I actually want", "my family", "sleep", "peace of mind",
    "the person I was before this",
]

# -------------------------------------------------------------------- DEEP
PATTERNS = [
    "shrink myself", "take on everyone else's problems", "sabotage things",
    "go silent", "over-explain", "say yes when I mean no",
    "leave before I get left", "work myself into the ground",
    "pick people who can't choose me", "need everyone to approve first",
    "quit right before it works", "make myself indispensable",
    "chase whoever is hardest to reach", "apologise for existing",
    "turn every relationship into a project", "keep score quietly",
    "wait to be picked", "prove myself to people who don't care",
    "abandon things at 80 percent", "make myself the strong one",
    "take the blame to end the conflict", "hide how much I want it",
    "confuse being needed with being loved", "test people until they leave",
    "outrun the feeling instead of sitting with it",
]
TRIGGERS = [
    "someone raises their voice", "things start going well",
    "I'm about to be seen", "there's a deadline",
    "someone offers me help", "I'm asked what I want",
    "authority is in the room", "I have to disappoint someone",
    "money is involved", "someone gets close",
]
PAST_INSTANCES = [
    "my first job", "my last relationship", "college placements",
    "the startup that failed", "my school years", "my previous manager",
    "the friendship that ended", "my first investor meeting",
    "the team I left", "the project I abandoned",
]
ROOTS = [
    "how my father handled disappointment",
    "being the child who never caused trouble",
    "the belief that love has to be earned",
    "growing up where mistakes were expensive",
    "always being told I was the responsible one",
    "watching my parents' silence after arguments",
    "being praised only for results",
    "learning early that needing things was inconvenient",
]
DEEP_VERBS = ["break", "understand", "unlearn", "sit with", "make sense of"]


def simple() -> str:
    kind = random.random()
    if kind < 0.35:
        text = random.choice(GREETINGS)
    elif kind < 0.6:
        text = random.choice(ONE_WORDERS)
    else:
        text = random.choice(DEFINITIONS)
    return (text + random.choice(SIMPLE_SUFFIX)).strip()


def normal() -> str:
    kind = random.random()
    if kind < 0.4:
        return f"I'm {random.choice(EMOTIONS)} about {random.choice(TOPICS)}."
    if kind < 0.7:
        return (
            f"My {random.choice(RELATIONSHIPS)} keeps "
            f"{random.choice(BEHAVIORS)} and I don't know what to do."
        )
    return (
        f"How do I {random.choice(ACTIONS)} when "
        f"{random.choice(SITUATIONS_SHORT)}?"
    )


def complex_() -> str:
    return (
        f"I've been {random.choice(LONG_SITUATIONS)} for "
        f"{random.choice(TIMEFRAMES)} and {random.choice(CONSEQUENCES)}. "
        f"How do I balance {random.choice(BALANCE_A)} with "
        f"{random.choice(BALANCE_B)}?"
    )


def deep() -> str:
    first, second = random.sample(PAST_INSTANCES, 2)
    return (
        f"I realize I always {random.choice(PATTERNS)} when "
        f"{random.choice(TRIGGERS)}. It happened with {first}, and again "
        f"with {second}, and I told myself both times it was just the "
        f"situation. I think it's connected to {random.choice(ROOTS)}. "
        f"I keep {random.choice(PATTERNS)} even when nothing is actually "
        f"at stake. How do I {random.choice(DEEP_VERBS)} this?"
    )


GENERATORS = {
    "simple": simple,
    "normal": normal,
    "complex": complex_,
    "deep": deep,
}


def main() -> None:
    rows = []
    for label, generator in GENERATORS.items():
        seen = set()
        attempts = 0
        while len(seen) < PER_CLASS and attempts < PER_CLASS * 60:
            attempts += 1
            seen.add(generator())
        if len(seen) < PER_CLASS:
            print(f"! {label}: only {len(seen)} unique variants possible")
        rows.extend({"query": q, "label": label} for q in seen)

    random.shuffle(rows)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ {len(rows)} examples -> {OUTPUT}\n")
    print(f"  {'label':<10}{'count':>7}{'avg words':>12}{'avg chars':>12}")
    for label in GENERATORS:
        subset = [r["query"] for r in rows if r["label"] == label]
        words = sum(len(q.split()) for q in subset) / max(1, len(subset))
        chars = sum(len(q) for q in subset) / max(1, len(subset))
        print(f"  {label:<10}{len(subset):>7}{words:>12.1f}{chars:>12.0f}")


if __name__ == "__main__":
    main()
