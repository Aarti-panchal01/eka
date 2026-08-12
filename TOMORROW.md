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

Live state at 23:02 (it moves ~26 pairs every minute, so re-read it):

| persona | on disk | target |
|---|---:|---:|
| founder | ~590 | 1000 |
| chanakya | 0 | 600 |
| gita | 0 | 600 |
| reflection | 0 | 1000 |

~2,600 pairs remain → **~1.6 h** at the measured rate. The queue and the watcher
are both already running, so if they survive the night this is finished before
you wake up and the watcher will have published it unattended.

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

2. **Rotate the exposed credentials.** Several were pasted in plaintext during
   the session: `GITHUB_TOKEN` (a classic PAT — likely repo read/write, the
   broadest one), plus the Groq, SambaNova, OpenRouter, Google, and Mistral keys.
   The HF token was already rotated once. Rotate the rest when convenient.

3. **The advice judge is a filter, not a proof.** `advice_regex_hit_rate: 0.0`
   means "no listed phrase appeared" — not "no advice present". The LLM judge
   catches 9/10 on adversarial phrasing; the last 10% needs a stronger judge
   model than 8B. Relevant only to reflection.

4. **The dataset has three teachers**, not one: Llama 3.3 70B (Groq), Nemotron 3
   Ultra (OpenRouter), Gemini 2.5 Flash (Google), and now Mistral Large. Every
   pair carries a `provider` field, so a single-teacher subset stays recoverable
   by filtering if the fine-tune shows voice inconsistency.

5. **Nemotron's output length is unstable.** Two identical-prompt batches gave
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
