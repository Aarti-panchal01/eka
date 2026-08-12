# Tomorrow — exact steps

State as of the end of the 2026-08-12 session. Read the **Reality check** section
before planning your day; the timeline is not what it looked like last night.

---

## 60-second version

```bash
cd ~/eka
bash scripts/morning_checklist.sh          # what finished, what's blocked
python ml/scripts/run_queue.py --all       # resume generation (quotas reset)
```

Then leave it. When all four datasets hit target, publish and start Kaggle:

```bash
python scripts/watch_and_publish.py        # waits, then preprocess + upload + Kaggle steps
```

---

## Reality check — read this first

**Generation is the only thing behind, and it is behind by days, not hours.**

| persona | on disk | target |
|---|---:|---:|
| founder | ~205 | 1000 |
| chanakya | 0 | 600 |
| gita | 0 | 600 |
| reflection | 0 | 1000 |

Last night's "all data ready tomorrow afternoon" estimate came from a
**2.5-minute throughput sample (216 pairs/hr)**. A later measurement over a
longer window gave ~76 pairs/hr including startup. The honest range is
**76–216 pairs/hr while Mistral is the only live provider**, and ~3,000 pairs
remain. Plan on **1–3 more days of generation**, and re-measure before trusting
any ETA. Do not build the day around the optimistic end.

**Why only Mistral:** every other free provider hit a hard daily wall.

| provider | status | real daily ceiling |
|---|---|---|
| mistral | **working** | meters per MONTH (~1B tokens) — effectively unlimited here |
| groq | out | 100,000 tokens/day (~133 pairs) |
| google | out | **20 requests/day** (~100 pairs) — requests bind, not tokens |
| openrouter | out | opaque; delivered ~112 pairs before stopping |
| github | **retired** | HTTP 410 `github_models_retirement_brownout` — gone, not throttled |

Groq, Google, and OpenRouter reset overnight, so tomorrow starts with four live
providers instead of one. That is the main reason to expect a better rate.

**The paid alternative, since it keeps coming up:** the Anthropic Batches API
does all 3,000 remaining pairs for roughly **$4.50**. Given how much time this
has absorbed, that is worth a second look before committing to more days of
free-tier rotation.

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
| Render | `render.yaml` complete; `infra/setup_render.md` written |
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

1. **Git has no remote and no commits.** I committed locally this session;
   `git push` needs a remote you create and credentials I don't have:
   ```bash
   git remote add origin git@github.com:amijackofalltrades/eka.git
   git push -u origin main
   ```
   Check `.env` never lands in a commit — it holds every key you own:
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
