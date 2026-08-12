# Tomorrow — exact steps

State as of the end of the 2026-08-12 session. Read the **Reality check** section
before planning your day; the timeline is not what it looked like last night.

---

## 60-second version

```bash
cd ~/eka
bash scripts/morning_checklist.sh          # what finished, what's blocked
```

**Read section 2 of that output before you type anything else.** The queue, the
watcher, and the backend were all left running overnight. If it says `queue
RUNNING`, do *not* start another — two queues share provider quotas and will
429 each other. Only if it says `stopped`:

```bash
python ml/scripts/run_queue.py --all       # resume generation (resume-safe)
python scripts/watch_and_publish.py        # waits, then preprocess + upload
```

Most likely branch: generation finished overnight and the watcher already
published to the Hub. Then you go straight to Kaggle — step 4.

---

## Reality check — read this first

**The 23:00 update supersedes everything written earlier tonight. Generation is
now roughly 1.5–2 hours from done, not 1–3 days.** The four-Mistral-key
round-robin changed the picture completely.

Measured at 23:00 on 2026-08-12, two independent ways that agree:

| method | window | rate |
|---|---|---|
| direct sample of `founder_dataset.json` | 5.8 min | **26.4 pairs/min** |
| `watcher.log` 2-min series, steady state | 6 min | **27.1 pairs/min** |

That is **~1,600 pairs/hr**, against the **76–216 pairs/hr** this file claimed an
hour earlier. The old number was measured when Mistral was a single key and the
only live provider. Do not plan against the old figure.

State at 00:07:

| persona | on disk | target | verdict |
|---|---:|---:|---|
| founder | **1000** | 1000 | **ok_to_train** |
| chanakya | 137 | 600 | generating |
| gita | 0 | 600 | queued |
| reflection | 0 | 1000 | queued |

**Rate is per-persona, and founder was the fast one.** Measured over a clean
4-minute window at 00:03, chanakya runs **13.2 pairs/min** against founder's
26. Nothing is wrong: chanakya's gates accept 51% of generated pairs where
founder's accept 71%, so it burns roughly twice the calls per kept pair. That
is the gate working, not a fault.

~2,060 pairs remain → **~2.6 h**, projecting a finish around **02:45**. Treat
that as a floor: reflection pays an LLM judge call per surviving pair and will
be slower than chanakya, so 3–4 h is the honest range. Queue and watcher are
both running, so this should finish and publish unattended overnight.

Do not re-derive an ETA from founder's 26/min — that number does not
generalise across personas.

**Three bugs stopped the queue tonight before any of this worked.** All three
are fixed and covered by tests; they are described under "What broke" below.
The short version: the queue stopped twice claiming quota exhaustion while the
Mistral keys were healthy, and neither stop was quota.

**Why it got fast:** four Mistral keys now round-robin, and Groq and OpenRouter
came back. Confirmed 8 distinct providers across 8 consecutive picks.

| provider | status | real daily ceiling |
|---|---|---|
| mistral ×4 keys | **working** | meters per MONTH (~1B tokens) — effectively unlimited |
| groq | back | 100,000 tokens/day (~133 pairs) |
| openrouter | back | opaque; ~112 pairs before it stopped last time |
| google | out | **20 requests/day** (~100 pairs) — requests bind, not tokens |
| github | **retired** | HTTP 410 `github_models_retirement_brownout` — gone, not throttled |

**On the paid shortcut:** earlier drafts of this file recommended spending ~$4.50
on the Anthropic Batches API to finish the remaining pairs. At 1,600 pairs/hr
that is no longer worth doing — the free rotation finishes tonight. Keep it in
your pocket only if the Mistral keys wall unexpectedly.

---

## What broke tonight, and what to watch for

Three separate faults, all of which presented identically — "out of quota" —
and none of which were quota. Worth knowing because the symptom is misleading.
A fourth item is a hardening change rather than a fault found.

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

**4. A near-miss would have blocked publishing overnight.** Not a fault found
tonight so much as one built out: any persona can end a pair or two short for
the reason described below, and the watcher will not publish a persona that is
not `ok_to_train`. The queue now retries a persona within 10 pairs of target,
once, suppressed entirely if anything stopped on quota.
Covered by `tests/test_queue_sweep.py`.

Run all four plus the E2E suite (the first four are fast and need no network;
E2E needs the backend up and takes ~70s):

```bash
python tests/test_throttle_waves.py
python tests/test_queue_outcome.py
python tests/test_queue_detection.py
python tests/test_queue_sweep.py
EKA_BASE_URL=http://127.0.0.1:8091 python tests/test_e2e.py
```

**Known but not chased: ~40% of generation calls fail to parse.** 184
unparseable replies against ~256 successful batches. `extract_pairs` already
salvages fenced code, whole arrays and individual balanced objects, so these
defeated all three. One reproduction returned a clean 5/5 at 2,582 tokens
against a 3,800 ceiling, so it is intermittent and not simple truncation. The
log line now reports length, a truncation flag and both ends of the reply, so
the next run diagnoses itself. Worth fixing only if there is more data to
generate — it costs throughput, not quality.

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

### 1. Check what happened overnight

```bash
bash scripts/morning_checklist.sh
```

Prints dataset counts, provider quota state, whether anything is still running,
and each persona's quality verdict. Start here — it tells you which of the
branches below you're in.

### 2. Resume generation

```bash
python ml/scripts/run_queue.py --all
```

Sequential by design (founder → chanakya → gita → reflection); they share
provider quotas, so parallel runs would 429 each other. Resume-safe — it reads
what's on disk and generates only the shortfall. Safe to re-run any number of
times.

It stops cleanly when every provider is walled and tells you so. That is the
expected ending, not a failure.

### 3. Watch for completion, then publish

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

### 4. Kaggle training

Only after step 3 uploads successfully. Full detail in
`scripts/start_kaggle_training.md`; the short version:

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

4. **The advice judge is a filter, not a proof.** `advice_regex_hit_rate: 0.0`
   means "no listed phrase appeared" — not "no advice present". The LLM judge
   catches 9/10 on adversarial phrasing; the last 10% needs a stronger judge
   model than 8B. Relevant only to reflection.

5. **The dataset has three teachers**, not one: Llama 3.3 70B (Groq), Nemotron 3
   Ultra (OpenRouter), Gemini 2.5 Flash (Google), and now Mistral Large. Every
   pair carries a `provider` field, so a single-teacher subset stays recoverable
   by filtering if the fine-tune shows voice inconsistency.

6. **Nemotron's output length is unstable.** Two identical-prompt batches gave
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
