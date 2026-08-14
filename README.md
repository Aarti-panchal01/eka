# Eka — Your Lifelong AI Companion

Four AI personas. Semantic memory that survives every conversation. Indian
voice, in your language.

**[Live →](https://the-eka.vercel.app)** · [API health](https://eka-backend-doau.onrender.com/health)

---

## What Eka does differently

Not a chat wrapper. The interesting part is everything between the question and
the model call.

- **Remembers across conversations** — semantic RAG over your own history, not
  a rolling context window
- **Routes by complexity** — four levels, and cheap questions never pay for
  deep retrieval
- **Speaks your language** — English, Hindi, Kannada, with per-persona voices
- **Four distinct minds**, each with its own dataset and adapter
- **Ranks memories by usefulness**, not just similarity

## The four personas

**⚡ Founder** — Brutally honest. Challenges assumptions, uses real startup
frameworks, ends on a sharp question.

**🛡 Chanakya** — Strategic and cold. Power dynamics, leverage, Arthashastra.
*"What is the one move available to you right now?"*

**✨ Gita** — Bhagavad Gita wisdom. Dharma, karma, detachment from outcomes.

**👁 Reflection** — CBT/ACT shaped. Never advises. Reflects, observes, asks.

---

## Architecture

```mermaid
flowchart LR
    U[message] --> C{complexity<br/>4 levels}
    C --> E[embed]
    E --> Q[(Qdrant)]
    Q --> R[LightGBM rerank<br/>NDCG@3 0.947]
    R --> P[persona prompt<br/>+ memories + goals]
    P --> L[Qwen2.5-7B]
    L --> O[reply]
    O --> M[(Postgres)]
    O --> T[Sarvam TTS]
    M -.write back.-> Q
```

### Retrieval budget scales with complexity

Most messages do not need deep retrieval. Paying for it anyway is what makes
these systems slow and expensive.

| Level | Memories | History turns | Candidate pool |
|---|---|---|---|
| Simple | 1 | 2 | 10 |
| Normal | 2 | 3 | 15 |
| Complex | 3 | 5 | 25 |
| Deep | 5 | 7 | 40 |

### Why a learned ranker

Cosine similarity returns what is *similar*, not what is *useful*. A memory
from yesterday usually beats a semantically closer one from a year ago, and
something you marked important beats both.

**LightGBM `LGBMRanker`, NDCG@3 = 0.9466** on held-out data, over seven
features: similarity, recency, priority, access count, length, mode match,
provenance. Reproducible with `python training/train_ranker_local.py`.

### Data pipeline — 3,200 pairs, quality-gated

| persona | pairs | train / val |
|---|---:|---:|
| founder | 1,000 | 900 / 100 |
| chanakya | 600 | 540 / 60 |
| gita | 600 | 540 / 60 |
| reflection | 1,000 | 900 / 100 |

Generated through a rotating pool of four free providers (Groq, Mistral ×4
keys, Google, OpenRouter) with per-key token pacing. Five gates plus an LLM
judge run on every pair: persona markers, response-length bounds,
one-question discipline, advice-leakage detection for `reflection`, and
near-duplicate rejection.

**The duplicate threshold is 0.75 Jaccard, and it is empirical, not chosen for
looking round.** At 0.75: 100% recall on clones, 0% false positives on the
hardest real collision. At 0.85 clone recall collapses to 21%.

Plus **6,000 embedding triplets** over 3,200 unique anchors, negatives drawn
from a *different* persona — exactly the confusion a retriever needs to stop
making.

### Voice

- **STT** — Sarvam `saarika:v2.5`, plus browser Web Speech for zero-latency dictation
- **TTS** — Sarvam `bulbul:v3`, per-persona speaker and pace:

| persona | speaker | pace | same line |
|---|---|---|---|
| founder | aditya | 1.10 | 4.51s |
| chanakya | ashutosh | 0.90 | 5.30s |
| gita | priya | 0.85 | 5.89s |
| reflection | kavya | 0.75 | 6.47s |

Questions are sent as their own request so they carry their own intonation.
Sentence breaks become `...` for real pauses — measured 2.48s → 3.6s on the
same two sentences.

---

## Evaluation

n=50 held-out prompts from `founder_val.jsonl` — the split that was actually
withheld from training. Four mechanical checks per response, no judge model.
`gold` scores the dataset's own answers, as an upper bound.

| Configuration | Persona score | Ends w/ question | Uses framework | No hedging | Words in band | Mean words |
|---|---|---|---|---|---|---|
| Training data (gold) | **94%** (3.74/4) | 78% | 100% | 96% | 100% | 189 |
| Base + prompt, no RAG | **93%** (3.73/4) | 100% | 100% | 100% | 73% | 159 |
| RAG + ranker (deployed) | **78%** (3.14/4) | 90% | 82% | 82% | 60% | 215 |
| Fine-tuned + RAG + ranker | _pending_ | — | — | — | — | — |

| Configuration | Memories retrieved | Precision@3 | Median latency |
|---|---|---|---|
| No RAG | — | — | **1,168 ms** |
| RAG + ranker | 4.7 | _needs labels_ | **16,796 ms** |

### What this actually says

**A base model with the persona prompt already scores 93% against gold's 94%.**
On these four checks, prompting is essentially at the ceiling — which sets a
real bar for the fine-tune to clear rather than assuming it helps.

**The two rows are not a clean ablation.** The no-RAG row runs
`llama-3.3-70b-versatile`; the deployed backend answers on
`llama-3.1-8b-instant`. So the 93→78 drop confounds *model size* with
*retrieval*, and the honest conclusion is "not yet measured", not "RAG hurts".
Fixing it means pinning both rows to the same model.

**These checks measure format, not substance.** The base model is *more*
mechanically compliant than gold — 100% of its answers end in a question
against gold's 78%. A fine-tune that copies the training distribution would
score slightly lower here and might still be better. Judging that needs blind
pairwise preference, which is the next thing to build.

**Precision@3 is deliberately blank.** Counting retrieved memories is not
knowing they were useful. `feedback_service` is now collecting implicit labels
from real sessions to fill it.

Reproduce: `python ml/eval/eval_harness.py`

## The ML work

### A validation pipeline, not just a generation script

Anyone can prompt an LLM into 3,200 rows. The work is refusing most of them.

Every generated pair passes five gates plus a judge before it can reach
training, and a persona that finishes short is regenerated rather than shipped:

- **Persona markers** — does the response actually behave like the persona
- **Length floors and ceilings** — 150–250 words, enforced
- **One-question discipline** — exactly one closing question, not three
- **Advice-leakage detection** — `reflection` must never give advice, and a
  regex misses paraphrase, so an 8B LLM judge classifies it semantically
- **Near-duplicate rejection** — Jaccard at **0.75**

**That threshold is derived, not chosen for looking round.** At 0.75: 100%
recall on clones, 0% false positives against the hardest real collision in the
corpus. At 0.85, clone recall collapses to **21%**. Jaccard is always the lower
of the two measures — for equal-sized sets, cosine 0.75 is Jaccard 0.60 — which
is why the number cannot be lifted from a paper that used cosine.

The gate refuses to publish anything not marked `ok_to_train`, because the next
step is a multi-hour GPU run that bakes any defect permanently into weights.

### Persona-conditioned retrieval

6,000 triplets over 3,200 unique anchors, and the interesting part is the
negatives: each one is drawn from a **different persona's** conversations.

That trains the retriever on the confusion that actually matters here. Generic
negatives teach "these two texts are unrelated", which cosine similarity
already knows. Persona negatives teach something harder — *the same topic,
discussed in a different register, is not the memory you want.* Founder talking
about failure and Gita talking about failure are semantically adjacent and
functionally opposite.

### A ranker, because similarity is the wrong objective

Cosine returns what is *similar*, not what is *useful*. A memory from yesterday
usually beats a closer one from a year ago; something the user flagged
important beats both.

**LightGBM `LGBMRanker`, NDCG@3 = 0.9466**, seven features: similarity,
recency, priority, access count, length, mode match, provenance. Trained,
deployed, and reproducible in one command.

Its honest weakness, stated because a reader will find it: it is trained on
synthetic data from `make_dataset()`. `feedback_service` is now collecting
implicit relevance labels from live sessions — which memories were retrieved,
and whether the user stayed on that thread — to retrain against real signal.

### An evaluation harness, and a result that cost something

Four mechanical checks over a held-out split, no judge model, fully
reproducible. It produced the most useful number in the project:

**Base model + persona prompt scores 93%. The training data itself scores 94%.**

Prompting is already at the ceiling of these metrics. That is a bar for the
fine-tune to clear, not an assumption to coast on — and finding it before
spending GPU hours is the entire point of building the harness first.

It also caught a confound in its own output: the RAG row runs a different base
model than the no-RAG row, so that comparison is reported as *not yet measured*
rather than as a finding.

### Training pipeline

Validated end to end on real hardware: 4-bit NF4 load, LoRA attach
(**40,370,176 trainable, 0.53% of 7.66B**), optimizer steps executing at a
measured **361 s/step** on a T4 before tuning.

Config, every value chosen from measurement: **r=8, α=32, 2 epochs, effective
batch 16, seq 1152**. That sequence length is not a guess — the longest example
in any split is 1,078 tokens, and the original 2,048 was pure padding cost, so
roughly half of every batch was pad tokens paid for at full price.

**No adapter has finished training yet.** Runs have been lost to a session cap
and a GPU-quota exhaustion, not to pipeline faults. When one completes it hot-
swaps into production behind the existing `LLM_MODE` switch, and the eval table
fills its pending row automatically.

## Status

| Component | State |
|---|---|
| Data pipeline — 3,200 gated pairs, 6,000 triplets | **done, on the Hub** |
| LightGBM ranker — NDCG@3 0.9466 | **trained, deployed, reproducible** |
| Evaluation harness + ablation table | **done, results published above** |
| Implicit relevance logging | **live, collecting** |
| Backend, RAG, voice, 5-screen UI | **done, deployed** |
| QLoRA training pipeline | **validated end to end on T4** |
| Persona adapters | **not yet complete** |
| Embedding / complexity / sentiment / summarizer | **scripts written, not yet trained** |

Personas today run from system prompts against Qwen2.5-7B via Groq, which is
why the product works end to end right now. Complexity and sentiment run
heuristic implementations — the free tier has 512MB, and `torch` plus
`transformers` resident does not fit alongside FastAPI.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React · Vite · Tailwind |
| Backend | FastAPI · SQLAlchemy async · Pydantic |
| Database | Supabase (PostgreSQL), 8 tables, alembic |
| Vector DB | Qdrant Cloud, 768-dim |
| LLM | Qwen2.5-7B-Instruct via Groq |
| Voice | Sarvam AI — Bulbul v3, Saarika v2.5 |
| Training | Kaggle / Colab T4, QLoRA via TRL + PEFT |
| Deploy | Vercel + Render |

## Run it

```bash
git clone https://github.com/Aarti-panchal01/eka && cd eka
cp .env.example .env                      # fill in keys
python scripts/verify_credentials.py      # checks every one for real

cd backend && pip install -r requirements.txt
uvicorn main:app --reload --port 8091
```

```bash
cd frontend && npm install
echo "VITE_EKA_API_URL=http://localhost:8091" > .env
npm run dev
```

Regenerate data or train:

```bash
python ml/scripts/run_queue.py --all       # generate all four personas
python ml/scripts/preprocess.py            # -> ChatML splits
python scripts/preflight_kaggle.py --hub   # refuses to waste a GPU session
python scripts/build_kaggle_notebooks.py   # notebooks are generated, not edited
```

---

## Things that cost real time

Written into the code as comments so they are not rediscovered:

- **Pacing is token-bound, not request-bound.** Going "faster" than the token
  budget made one generation run 6 days instead of 6 hours.
- **A trailing comma discarded ~40% of generated batches.** `...right now?", } ]`
  is invalid strict JSON, so a whole five-pair call was thrown away.
- **Never pin a stale stack against a managed image.** `numpy<2.0` on a host
  built around NumPy 2 produced a mixed-ABI crash one cell after the pin
  appeared to succeed.
- **Verify the artifact, not the exit code.** A training notebook can finish
  cleanly having pushed nothing at all.
- **A field that never enters the app cannot leak.** Routing internals are
  stripped in the API client, not hidden in the view.

---

## Built by

**Aarti Panchal** — B.Tech AI & ML, PES University Bengaluru
C4GT DMP '26 Fellow

[LinkedIn](https://linkedin.com/in/aarti-panchal) ·
[GitHub](https://github.com/Aarti-panchal01) ·
[Portfolio](https://aarti-tech-portfolio.vercel.app)

---

*Eka is Sanskrit for "one" — the one companion that knows you completely.*
