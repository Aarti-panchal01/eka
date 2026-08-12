# Eka — LLM serving: the decision guide

Phase 7 wiring is done (`serving/merge_lora_and_serve.py`,
`serving/hf_space_app.py`). This file is the "which option do I actually use,
and how do I know it's working" guide that ties them together with
`backend/config.py` and `backend/services/llm_service.py`.

There is no Oracle VM in this build — that plan was dropped. Every option
below runs on either your own machine, a free managed platform, or Groq's
hosted API.

## The honest comparison

| | Groq (default) | Local Ollama + ngrok | HF Space (ZeroGPU) | Paid GPU host |
|---|---|---|---|---|
| **Personas fine-tuned?** | No — base `llama-3.1-8b-instant`, not your LoRA | Yes — your merged `eka-<mode>` models | Yes — base + your PEFT adapters | Yes |
| **Cost** | Free | Free (your electricity + time) | Free (ZeroGPU quota) or Pro | $ (Lambda/RunPod/etc, ~$0.5-2/hr for an 8B model) |
| **Setup effort** | Zero — already wired, just needs `GROQ_API_KEY` | Merge + register + tunnel (this doc) | Deploy a Space, load adapters | Rent a box, deploy a server |
| **Needs your machine on?** | No | **Yes**, continuously, or the backend silently falls back to Groq | No | No |
| **URL stability** | Stable (Groq's API) | **Churns** — free ngrok gives a new URL every restart | Stable (`*.hf.space`) | Stable |
| **Cold start** | None | None once Ollama is warm | Yes — ZeroGPU spins a GPU up per call, and this app lazy-loads the model+adapters on first request (expect 20-60s the first time) | Depends on autoscaling config |
| **Queueing** | Groq's own rate limits | None (single machine, single request at a time in practice) | Yes — ZeroGPU is a shared, quota-limited pool; busy times mean waiting | None (dedicated) |
| **Latency once warm** | Fast (Groq's inference is very fast) | Depends on your CPU/GPU — can be slow on CPU-only | Fast once GPU is granted | Fast |
| **Best for** | Getting the app working today, demos, judging base capability | Real persona demos when you're at your machine | Real persona demos without babysitting a machine, sharing a live link | Production, if this ever needs to run unattended 24/7 |

**Bottom line:** start on Groq because it already works. Switch to Ollama+ngrok
for a live demo where you control the machine. Switch to the HF Space when
you want a link you can share that doesn't depend on your laptop being on.
Only look at paid hosting if this needs to be always-on and fast for real
users.

## The env-var switch

All three modes are read from `backend/config.py` (`Settings.LLM_MODE`). Set
whichever one you're using in `.env` (local) or the Render dashboard
(deployed):

```bash
# Option 0 — Groq (default, works with zero other setup)
LLM_MODE=groq
GROQ_API_KEY=gsk_...

# Option A — local Ollama, or Ollama tunnelled with ngrok
LLM_MODE=ollama
OLLAMA_BASE_URL=http://localhost:11434          # local
OLLAMA_BASE_URL=https://<random>.ngrok-free.app  # via ngrok, from Render

# Option B — Hugging Face Space (ZeroGPU)
LLM_MODE=hf_space
HF_SPACE_URL=https://<you>-eka-personas.hf.space
HF_TOKEN=hf_...                                  # only if the Space requires auth
```

No code changes are needed to switch — `llm_service.py` reads `LLM_MODE` at
request time and always keeps Groq as the last-resort fallback, regardless
of which primary mode you pick.

## Option A — local Ollama + ngrok, step by step

1. Train the adapter(s) if you haven't: `training/train_<mode>_lora_kaggle.py`
   (on Kaggle) pushes to `<HF_USERNAME>/eka-<mode>-lora` on the Hub.
2. Run the one-command driver from the project root:
   ```bash
   python serving/merge_lora_and_serve.py --mode founder
   # or, for all four:
   python serving/merge_lora_and_serve.py --all
   ```
   This merges the adapter (shells out to `ml/scripts/merge_lora.py`),
   registers it with Ollama (`ollama create eka-founder -f Modelfile`), and
   verifies it responds to `ollama run eka-founder "hello"`.
3. To also expose it to a remote Render backend:
   ```bash
   python serving/merge_lora_and_serve.py --mode founder --ngrok
   ```
   This starts `ngrok http 11434` and prints the public HTTPS URL plus the
   exact env vars to paste into Render.
4. In the Render dashboard, set `LLM_MODE=ollama` and
   `OLLAMA_BASE_URL=<the ngrok URL>`, then redeploy/restart the service so it
   picks up the new env vars.
5. Keep the terminal running `ngrok http 11434` open, and keep Ollama running
   (`ollama serve`, or the Ollama desktop app), for as long as you want the
   deployed backend to reach your machine. Closing either one drops LLM_MODE
   back to the Groq fallback silently (see the debugging section below for
   how to notice that happened).

Re-running the ngrok tunnel gives you a **new URL** every time (unless you
pay for a reserved ngrok domain) — you must update `OLLAMA_BASE_URL` in
Render again after every restart.

## Option B — Hugging Face Space (ZeroGPU), step by step

Full instructions (requirements.txt, README frontmatter, Dockerfile, repo
secrets) are in the comment block at the bottom of `serving/hf_space_app.py`.
Summary:

1. Create a Space at https://huggingface.co/new-space (Docker or Gradio SDK
   — see the note in `hf_space_app.py` about which SDKs support ZeroGPU on
   your account tier).
2. Push `hf_space_app.py`, its `requirements.txt`, `README.md` (with the
   frontmatter block), and a `Dockerfile` (if using `sdk: docker`) to the
   Space repo.
3. Add repo secrets: `HF_TOKEN` (needs read access to your adapter repos and
   the gated Llama-3 base model — accept its license on huggingface.co
   first) and `HF_USERNAME`.
4. Space Settings -> Hardware -> ZeroGPU.
5. Once built, `GET https://<you>-<space>.hf.space/health` — `model_loaded`
   stays `false` until the first `/generate` call (lazy load, so idle cost is
   low).
6. Set `LLM_MODE=hf_space` and `HF_SPACE_URL=https://<you>-<space>.hf.space`
   in the backend's env.

This Space design loads the base Llama-3-8B **once** and layers all four
persona LoRA adapters on top (`PeftModel` + `load_adapter`/`set_adapter`), so
`mode` in the request body picks a persona without reloading anything from
disk. See the docstring at the top of `hf_space_app.py` for the full
rationale and the one-Space-per-persona alternative it considered.

## How to verify which backend is *actually* answering

This is the single most useful debugging habit for this setup, because every
backend silently falls back to Groq on failure — the app never errors, it
just quietly serves base-model answers instead of your fine-tuned personas.

1. **`GET {BASE_URL}/health`** — returns (from `backend/main.py`, backed by
   `LLMService.health_check()`):
   ```json
   {
     "status": "ok",
     "llm_mode": "ollama",
     "ollama": true,
     "groq": true,
     "hf_space": false,
     ...
   }
   ```
   `llm_mode` is what's *configured*. `ollama` / `groq` / `hf_space` are
   whether each backend is currently *reachable* — not which one actually
   served the last reply. If `llm_mode=ollama` but `ollama: false`, your
   tunnel or local Ollama is down and every request is quietly going to Groq.

2. **The `degraded` array on a real `/chat/send` response** — this is the
   precise, per-request signal. `services/rag_service.py` builds this list
   and includes an entry shaped exactly like:
   ```
   "degraded": ["llm:ollama->groq"]
   ```
   whenever `LLM_MODE` is `ollama` or `hf_space` but the request actually got
   answered by the Groq fallback. `ChatResponse.llm_backend` in the same
   response tells you which backend produced *this specific* reply
   (`"ollama"`, `"groq"`, `"hf_space"`, or `"none"` if everything failed).
   No `degraded` entry + `llm_backend` matching your configured `LLM_MODE`
   means your fine-tuned persona really is answering.

   ```bash
   curl -s -X POST "$BASE/chat/send" \
     -H "Content-Type: application/json" \
     -d '{"user_id":"debug","mode":"founder","message":"hello"}' \
     | python -m json.tool
   ```
   Check `.llm_backend` and `.degraded` in the output.

3. Combine both: `/health` tells you what's reachable right now; `/chat/
   send`'s `llm_backend` + `degraded` tell you what actually served the last
   real request. If they disagree (e.g. `/health` says `ollama: true` but a
   chat response still shows `llm:ollama->groq`), the model that answered
   may be the wrong persona (see "model not found" below) or the probe and
   the actual request raced against a flaky tunnel.

## Troubleshooting

- **"model not found" / wrong persona answers** — `ollama_model_for_mode()`
  in `config.py` maps `mode` to `eka-<mode>`. If that exact model isn't
  registered, `llm_service._ollama()` logs a warning and silently substitutes
  *any* other registered Ollama model rather than failing — so you'll get a
  reply, just from the wrong persona (or the base model). Run
  `ollama list` to see what's actually registered, and re-run
  `serving/merge_lora_and_serve.py --mode <mode>` for the missing one.

- **ngrok URL changed** — free ngrok tunnels get a new random URL on every
  restart. If Render's `OLLAMA_BASE_URL` still points at yesterday's URL,
  `/health` will show `ollama: false`. Re-run
  `serving/merge_lora_and_serve.py --mode founder --ngrok` (or `--all
  --skip-merge --ngrok` if already merged) to get a fresh URL, and update
  Render's env var again.

- **Space cold start / timeout** — the first `/generate` call after a Space
  wakes (or after a long idle period) has to load the base model + adapters,
  which can take 20-60s, on top of ZeroGPU's own queueing for a free GPU
  slot. `LLM_TIMEOUT_SECONDS` in `config.py` (default 60s) may need raising
  if you see `hf_space` failing with a timeout right after a cold Space —
  or just accept the fallback to Groq for that one request and let the next
  one hit a now-warm Space.

- **401 on the gated Llama-3 repo** — `meta-llama/Meta-Llama-3-8B-Instruct` is
  gated. Visit its page on huggingface.co and accept the license with the
  *same account* whose `HF_TOKEN` you're using, or every merge/load step
  (locally, in `merge_lora_and_serve.py`, and in the Space) will fail with a
  401/403 fetching the base model.

- **Out-of-memory during merge** — `ml/scripts/merge_lora.py` loads the full
  fp16 8B base model on CPU, which needs ~16GB RAM. If it gets OOM-killed:
  close other memory-heavy applications, merge one mode at a time (never
  `--all` on a constrained machine), or add swap. `serving/
  merge_lora_and_serve.py` checks free *disk* space before merging but cannot
  check free RAM portably — watch for the process dying silently or with a
  `Killed` message if you're tight on memory.

- **`ollama create` fails with a cryptic error** — usually either the
  Modelfile's `FROM` path is wrong (check it points at `./` for safetensors
  or at the right `.gguf` filename after `--gguf` conversion) or the Ollama
  daemon isn't running at all (`ollama serve`). `serving/
  merge_lora_and_serve.py` prints the full stderr from the failed `ollama
  create` call — read it, it's almost always specific about which file it
  couldn't find.

- **huggingface_hub can't confirm the adapter repo exists** —
  `merge_lora_and_serve.py`'s prerequisite check needs `huggingface_hub`
  installed and a valid `HF_TOKEN`/`HF_USERNAME`. If the check itself errors
  out (rather than cleanly reporting "not found"), it prints a warning and
  proceeds optimistically — don't assume the merge will actually succeed in
  that case; check the repo on huggingface.co directly if the merge step
  then fails.
