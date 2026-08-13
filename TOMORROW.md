# Tomorrow — exact steps

**Generation is DONE. All four datasets are on the Hub. Go to Kaggle.**

Finished 03:02 on 2026-08-13. The watcher preprocessed, gated and published
unattended, exactly as designed.

| persona | pairs | train / val | verdict |
|---|---:|---:|---|
| founder | 1000/1000 | 900 / 100 | **ok_to_train** |
| chanakya | 600/600 | 540 / 60 | **ok_to_train** |
| gita | 600/600 | 540 / 60 | **ok_to_train** |
| reflection | 1000/1000 | 900 / 100 | **ok_to_train** |

2,880 train / 320 val, longest sequence ~1,101 tokens against a 2,048 ceiling.
`marker=1.0` and `unique=True` on all four. Live at
`huggingface.co/datasets/amijackofalltrades/eka-datasets` (private).

**Embedding triplets are generating** (started 06:46, ~2.5 h for round 1).
They were the one thing blocking Kaggle **session 5**; sessions 1–4 and 6
never needed them and are ready now. When it finishes, re-push and confirm:

```bash
python ml/scripts/upload_to_hf.py
python scripts/preflight_kaggle.py --hub
```

Progress without touching the API: `grep -c . ml/datasets/embedding_triplets.jsonl`

---

## 60-second version

```bash
cd ~/eka
python scripts/preflight_kaggle.py --hub    # expect: 4/4 ready, exit 0
```

That was green at 03:03. If it still is, **start Kaggle session 1 (founder)** —
jump to step 4. Nothing else needs doing first.

Only if something looks wrong:

```bash
bash scripts/morning_checklist.sh           # full state, no API calls
```

Read section 2 before starting anything: if it says `queue RUNNING`, do *not*
start another — two queues share provider quotas and will 429 each other.

---

## How it finished, in case you need to do this again

Generation took ~4 hours for 3,200 pairs, not the 1–3 days this file predicted
at 22:00. Four things mattered, in rough order of impact:

**Four Mistral keys round-robining.** This was the single biggest change.
Measured rates: founder 26 pairs/min, chanakya 13, reflection 31. Rate is
**per-persona, not global** — it tracks gate acceptance (founder 71%, chanakya
51%), so do not extrapolate one persona's number to the rest. That mistake
produced two wrong ETAs tonight.

**A trailing comma was discarding ~40% of all calls.** Replies ended
`...right now?", } ]`, which is invalid strict JSON, so `json.loads` rejected
the whole array and the balanced-object fallback failed identically. One stray
comma threw away all five pairs in a call. Fixed in `extract_pairs`, and the
effect was immediate: chanakya's last 8 and gita's last 14 — the hardest pairs
in the run — closed within two minutes of the fix going live, after hours of
not closing.

**Every persona finishes short on its first pass.** ~12% attrition, because a
generator plans enough batches to cover the gap once and has nothing left when
the gates take their share. The queue now sweeps short personas repeatedly
until the gap stops closing.

**Reflection was the fastest, not the slowest.** Its LLM judge runs on an 8B
model, batched, on a separate provider pool with its own rate limiter — the
design deliberately keeps it off the critical path, and it works.

**Provider reality as of tonight:**

| provider | status | real daily ceiling |
|---|---|---|
| mistral ×4 keys | **working** | meters per MONTH (~1B tokens) — effectively unlimited |
| groq | per-model limits | 70B walls early; **8B still answers**, which is what keeps the judge alive |
| openrouter | walls early | opaque; ~112 pairs before it stopped |
| google | out | **20 requests/day** — requests bind, not tokens |
| github | **retired** | HTTP 410 `github_models_retirement_brownout` — gone, not throttled |

The Anthropic Batches API shortcut earlier drafts recommended is moot — the
free rotation finished the job in one night.

---

## What broke tonight, and what to watch for

Five items. The first three presented identically — "out of quota" — and none
of them were quota, which is the part worth remembering: the symptom pointed
away from the cause every time. Item 4 is a hardening change, and item 5 was
the single biggest throughput win of the night.

**1. A 2-second throttle read as a daily wall.** `complete()` bounds each call
at `max_rotations` attempts. Six workers against four keys burst past the
per-key rate, and rotating costs no wall time, so the budget drained in
seconds and returned the same "nothing available" signal a real daily wall
returns. The caller turns that into a fatal stop. Killed the run at 844/1000
with 213k of 6,000,000 Mistral tokens spent. Now a call waits the throttle out
(5s, 10s, … capped at 30s, six waves) when anything is still available.
Covered by `tests/test_throttle_waves.py`.

**2. One un-generatable pair blocked three personas.** `QUOTA_MARKERS` was five
phrases, four of which print on the normal healthy path whenever a single
secondary provider parks. founder ended at 999/1000 — one pair short — and the
substring scan read the routine chatter as a fatal stop, so chanakya, gita and
reflection never started. 2,200 pairs blocked by one.
Covered by `tests/test_queue_outcome.py`.

**3. The watcher could not see the queue die.** `queue_running()` asked whether
any `python.exe` existed; the watcher is one, so it always matched itself. The
"queue stopped early" branch was unreachable. The queue died at 23:20 and the
watcher polled on for 40 minutes reporting nothing wrong.
Covered by `tests/test_queue_detection.py`.

**4. Every persona finishes short, and that would have blocked publishing.**
Not a fault in the code so much as a fact about the generators: they plan
enough batches to cover the gap once, and have no budget left when the gates
take their share. chanakya finished **529/600** with a 0.42 rejection rate and
only 0.34 of those recovered by regeneration — roughly 12% attrition — while
its report was otherwise spotless (marker 1.0, zero near duplicates, 99.6%
ends-with-question). founder finished 999/1000. Any shortfall makes the verdict
`needs_review`, and the watcher will not publish a persona that is not
`ok_to_train`, so all three would have sat on disk until morning.

The queue now sweeps any short persona, repeatedly, while passes are still
closing the gap — bounded at three, exiting the moment a pass gains nothing,
and suppressed entirely by a quota or error stop. A fresh run regenerates the
remaining gap from scratch, which is how founder went 999 → 1000 on one extra
call and chanakya went 529 → 583 on one sweep.

The first version of this only retried personas within 10 pairs of target,
sized on founder's 1-pair case before chanakya had finished. That gate would
have skipped every persona it was written to rescue.
Covered by `tests/test_queue_sweep.py`.

**5. A trailing comma was discarding ~40% of all generation calls.** 184
unparseable replies against ~256 successful batches. `extract_pairs` already
salvaged fenced code, whole arrays and individual balanced objects, and these
defeated all three — because every one of them ended
`...right now?", } ]`. A trailing comma is invalid strict JSON, so
`json.loads` rejected the entire array and the balanced-object fallback
re-parsed the same text and failed identically. One stray comma threw away all
five pairs in a call.

Nothing was truncated: the failures ran 1,900–5,800 chars and closed cleanly
with `]` and a fence. The model simply writes JSON5-ish output.

It was invisible until the unparseable-reply log started printing both ends of
the failing text — before that, a trailing comma, a chatty preamble and a
refusal were indistinguishable. The repair is string-aware rather than a
regex, since these are long responses full of commas.
Covered by `tests/test_extract_pairs.py`.

The effect was immediate and is the clearest evidence it was the real cause:
chanakya's last 8 pairs and gita's last 14 — the hardest draws in the run,
stuck for hours against saturated topics — closed within two minutes of the
fix going live.

Run all five plus the E2E suite (the first five are fast and need no network;
E2E needs the backend up and takes ~70s):

```bash
python tests/test_throttle_waves.py
python tests/test_queue_outcome.py
python tests/test_queue_detection.py
python tests/test_queue_sweep.py
python tests/test_extract_pairs.py
EKA_BASE_URL=http://127.0.0.1:8091 python tests/test_e2e.py
```


**The last pair of a persona is a genuinely hard draw — this is not a bug.**
founder sat at 999/1000 across several runs. 14 of its 15 topics were exactly
full; only `investor relationship` was 59/60, so the final pair had to be about
that topic *and* unlike the 59 already there. The dedup gate was doing its job.
Forcing it would mean admitting a near-duplicate, which is worse than being one
pair short. It cleared on a later attempt with no threshold changed. If a
persona parks one short, just run its generator again:

```bash
python ml/scripts/generate_founder_data.py     # or chanakya / gita / reflection
```

Do **not** relax `_gen_quality.py`'s shortfall check to get past this. It is the
only thing standing between a near-duplicate and a multi-hour GPU run.

---

## Everything else is done and verified

Nothing below needs work tomorrow. Verified this session, not assumed:

| area | state |
|---|---|
| Supabase schema | **all 8 tables present**, alembic at `002_indexes` |
| E2E suite | **7/7 passing** against the live backend |
| Backend services | no TODO / FIXME / NotImplemented anywhere in `backend/` |
| Frontend client | `frontend/src/api/ekaClient.js`, 692 lines, ~40 methods, covers every route group |
| HF Hub repos | all **9** exist and are private under `amijackofalltrades` |
| HF auth | token rotated and verified (`whoami` → `amijackofalltrades`) |
| Kaggle scripts | 4 persona LoRA runs, each with checkpoint-resume + secret instructions |
| Kaggle notebooks | 4 `.ipynb` in `ml/notebooks/`, generated from `training/`, nbformat-valid |
| Render | `render.yaml` verified against `backend/config.py` — every setting needing a value on Render is declared, no typo'd keys |
| Backend deps | 4 advisories patched (python-multipart, lightgbm, python-dotenv, PyPDF2→pypdf); E2E 7/7 after the bump |
| Quality gates | 5 gates + LLM advice judge, all unit-verified |

---

## Step by step

**Steps 1–3 already ran and succeeded overnight. They are kept for the day you
need to generate data again — skip to step 4.**

### 1. ~~Check what happened overnight~~ (done 03:02)

```bash
bash scripts/morning_checklist.sh
```

Prints dataset counts, provider quota state, whether anything is still running,
and each persona's quality verdict. Start here — it tells you which of the
branches below you're in.

### 2. ~~Resume generation~~ (done — all four at target)

```bash
python ml/scripts/run_queue.py --all
```

Sequential by design (founder → chanakya → gita → reflection); they share
provider quotas, so parallel runs would 429 each other. Resume-safe — it reads
what's on disk and generates only the shortfall. Safe to re-run any number of
times.

It stops cleanly when every provider is walled and tells you so. That is the
expected ending, not a failure.

### 3. ~~Watch for completion, then publish~~ (done — published 03:02)

```bash
python scripts/watch_and_publish.py
```

Polls every 2 minutes; when all four reach target it runs `preprocess.py`, gates
on every persona's quality report, then `upload_to_hf.py`, then prints the Kaggle
steps.

**It will refuse to upload a dataset whose verdict is `needs_review`.** That is
deliberate: the next step is a multi-hour GPU run that bakes any defect
permanently into weights.

To publish at 90% rather than waiting for the last few pairs:

```bash
python scripts/watch_and_publish.py --min-frac 0.9
```

### 4. Kaggle training  ← **START HERE**

Only after step 3 uploads successfully. **Run the pre-flight first — one
second, and it is the difference between finding a problem now and finding it
15 minutes into a GPU session, after the 16GB model download:**

```bash
python scripts/preflight_kaggle.py --hub
```

Exit 0 means every persona has valid splits, full Llama-3 scaffolding, a
non-empty val set, nothing over `MAX_SEQ_LEN`, an `ok_to_train` verdict, and
the files are actually on the Hub where Kaggle will look for them.

Full detail in `scripts/start_kaggle_training.md`; the short version:

**Upload the notebook, don't paste the script.** `ml/notebooks/` now holds one
`.ipynb` per persona:

```
ml/notebooks/eka_founder_lora_kaggle.ipynb
ml/notebooks/eka_chanakya_lora_kaggle.ipynb
ml/notebooks/eka_gita_lora_kaggle.ipynb
ml/notebooks/eka_reflection_lora_kaggle.ipynb
```

Kaggle → **New Notebook → File → Import Notebook**. Each is the matching
`training/*.py` split into one cell per section, so a failure gives you a
traceback pointing at a named stage, and you can re-run just the push-to-Hub
cell if a 3-hour train succeeds and the upload 401s. Pasting the whole script
into one cell still works — it is the same code — you just lose both of those.

These are **generated**. Edit `training/train_*.py`, then:

```bash
python scripts/build_kaggle_notebooks.py           # rebuild all four
python scripts/build_kaggle_notebooks.py --check   # fails if a notebook is stale
```

- Session 1 = **founder**. Watch this one to completion — every mistake you're
  going to make (unattached secret, unaccepted Llama license, missing split)
  surfaces here.
- Sessions 2–4 = chanakya, gita, reflection. Same procedure, different file.
- **Always "Save Version → Save & Run All (Commit)"**, never the interactive Run
  All button — the latter dies when you close the tab, and these are 3-hour runs.
- GPU quota is 30 h/week. Four runs at ~3.5 h worst case is ~14 h, so all four
  fit in one week with room to retry.

---

## Known issues, none blocking

1. **Pushed — the repo is PRIVATE.**
   [github.com/Aarti-panchal01/eka](https://github.com/Aarti-panchal01/eka),
   branch `main`, 2 commits. Private was chosen deliberately: the history is
   clean, but this is unreleased work and flipping to public later is one
   click, whereas the reverse is not. Make it public whenever you want:
   ```bash
   gh repo edit Aarti-panchal01/eka --visibility public
   ```
   Note the account is `Aarti-panchal01` (your GitHub login), not
   `amijackofalltrades` (your Hugging Face login) — earlier drafts of this file
   assumed the two matched.

   Verified before pushing: no key-shaped string in the worktree *or* in the
   committed history, and `.env` untracked. Re-check any time:
   ```bash
   git ls-files | grep -c '^\.env$'      # must print 0
   ```

2. **~60 Dependabot alerts remain, all torch/transformers, none deployed.**
   GitHub reports 66 total. The four that reach the running backend are fixed
   (see the table above). The rest are `torch` and `transformers` in
   `ml/requirements.txt` and `serving/requirements.txt` — `render.yaml`
   deliberately leaves both uninstalled on the free tier, and the serving
   services are commented out, so nothing deployed imports them.

   They are **deliberately not bumped**: the Kaggle notebooks pin
   `transformers==4.41.0` / `torch==2.3.0` inline, that combination is what the
   training scripts were calibrated against, and changing it hours before four
   3-hour GPU runs trades a theoretical risk for a real one. The critical
   `torch.load` advisory needs untrusted weights to matter; these runs load
   `meta-llama/Meta-Llama-3-8B-Instruct` from the Hub. Revisit after the
   adapters exist.

3. **Rotate the exposed credentials.** Several were pasted in plaintext during
   the session: `GITHUB_TOKEN` (a classic PAT — likely repo read/write, the
   broadest one), plus the Groq, SambaNova, OpenRouter, Google, and Mistral keys.
   The HF token was already rotated once. Rotate the rest when convenient.

4. **The advice judge is down to ONE live provider.** Probed at 01:47, before
   reflection started, because reflection is the only persona that pays a judge
   call per surviving pair and a dead judge there costs an hour to discover.

   Result: the judge works — it returned the right verdicts on an
   advice-giving and a reflective sample. But of its four configured
   providers only two have keys, and one of those is now gone:

   | judge provider | state |
   |---|---|
   | `groq-8b` | **live** — and answering while Groq's 70B generation model is daily-walled, because Groq meters per model |
   | `sambanova-8b` | **HTTP 410 — `Meta-Llama-3.1-8B-Instruct` is deprecated on SambaNova Cloud** |
   | `cerebras-8b` | no `CEREBRAS_API_KEY` |
   | `together-8b` | no `TOGETHER_API_KEY` |

   Fine for tonight: Groq's 8B allows ~14,400 req/day against reflection's
   ~1,000. But there is no fallback left, so if Groq's 8B walls mid-run the
   gate is lost. To restore redundancy, point `SAMBANOVA_JUDGE_MODEL_ID` at a
   model SambaNova still serves, or add `CEREBRAS_API_KEY`. Do not simply
   reuse the Mistral generation keys — the judge pool is separate on purpose
   so judging cannot eat the generation model's per-minute budget.

5. **The advice judge is a filter, not a proof.** `advice_regex_hit_rate: 0.0`
   means "no listed phrase appeared" — not "no advice present". The LLM judge
   catches 9/10 on adversarial phrasing; the last 10% needs a stronger judge
   model than 8B. Relevant only to reflection.

6. **The dataset has three teachers**, not one: Llama 3.3 70B (Groq), Nemotron 3
   Ultra (OpenRouter), Gemini 2.5 Flash (Google), and now Mistral Large. Every
   pair carries a `provider` field, so a single-teacher subset stays recoverable
   by filtering if the fine-tune shows voice inconsistency.

7. **Nemotron's output length is unstable.** Two identical-prompt batches gave
   5/5 (user 142–155w) and 0/5 (user 76–95w). The gates catch it; it just costs
   throughput.

---

## Useful commands

```bash
# where everything stands, no API calls
python ml/scripts/run_queue.py --status

# which providers are alive right now
python ml/scripts/_gen_providers.py --check

# re-score a dataset against current gates, no API calls
python ml/scripts/generate_founder_data.py --report-only

# cost/time estimate before committing to a run
python ml/scripts/generate_founder_data.py --plan

# E2E against the running backend
EKA_BASE_URL=http://127.0.0.1:8091 python tests/test_e2e.py

# restart the backend (picks up .env changes; it reads keys once at startup)
cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8091
```
