# Deploying Eka's backend to Render

Step-by-step, in the order you actually do it. Everything here assumes the
repo already contains `render.yaml` at the root (it does) and that the backend
lives in `backend/` (it does).

There are two routes. **Blueprint** reads `render.yaml` and creates the
service for you — fewer clicks, fewer mistakes, and it is what this repo is
built for. **Manual** is the click-through path the Render docs describe; use
it only if the Blueprint page errors out.

---

## Before you start — have these five values in hand

The deploy will stop and ask for them. Get them out of your local `.env`
first so you are not hunting mid-deploy.

| Value | Where it comes from |
|---|---|
| `DATABASE_URL` | Supabase → Project Settings → Database → Connection string → **Session pooler** |
| `GROQ_API_KEY` | https://console.groq.com/keys |
| `QDRANT_URL` + `QDRANT_API_KEY` | Qdrant Cloud → your cluster → API keys |
| `HF_TOKEN` | https://huggingface.co/settings/tokens (needs *Inference Providers* permission) |
| `SARVAM_API_KEY` | https://dashboard.sarvam.ai |

Plus a `SECRET_KEY` — generate a fresh one, do not reuse the local dev value:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### Two `DATABASE_URL` details that will bite you

1. **Use the session pooler host** (`...pooler.supabase.com`), not the direct
   `db.<ref>.supabase.co` host. Render's free tier is IPv4-only; the direct
   host is IPv6-only and will simply fail to connect.
2. **URL-encode the password.** If it contains `@`, write it as `%40`,
   otherwise the connection string parser reads the `@` as the host separator
   and the URL breaks. Other characters that need encoding: `:` → `%3A`,
   `/` → `%2F`, `?` → `%3F`, `#` → `%23`, `%` → `%25`.

You do not need to add the `+asyncpg` driver yourself — `backend/config.py`'s
`_force_async_driver` validator rewrites `postgresql://` to
`postgresql+asyncpg://` on load. Pasting Supabase's string as-is is fine.

---

## Route A — Blueprint (recommended)

1. Push the repo to GitHub if you have not. **This repo currently has zero
   commits**, so this is a real step, not a formality:

   ```bash
   cd ~/eka
   git add -A
   git commit -m "Eka backend, ML pipeline, and deploy config"
   git remote add origin git@github.com:<you>/eka.git
   git push -u origin main
   ```

   Confirm `.env` did **not** get committed — `.gitignore` covers it, but
   check, because every secret you own is in that file:

   ```bash
   git ls-files | grep -c '^\.env$'    # must print 0
   ```

2. Go to **https://dashboard.render.com/blueprints** → **New Blueprint
   Instance**.

3. **Connect your GitHub account** if prompted, then pick the `eka` repo.
   Render reads `render.yaml` and shows you one service to create:
   `eka-backend`.

4. Render lists every var marked `sync: false` as a blank field. Fill in the
   five secrets above plus `SECRET_KEY`. Leave these as-is — `render.yaml`
   already sets them and they are correct for the free tier:

   - `COMPLEXITY_SERVE_URL`, `RANKER_SERVE_URL`, `SENTIMENT_SERVE_URL` — empty
   - `ENABLE_LOCAL_CLASSIFIERS` — `false`

   Set `FRONTEND_URL` and `CORS_ORIGINS` to your deployed frontend origin. If
   the frontend is not up yet, put `*` in `CORS_ORIGINS` and come back and
   tighten it — do not ship `*` permanently.

   Optional but useful: `GROQ_GEN_MODEL=llama-3.3-70b-versatile`,
   `WANDB_API_KEY` if you want training runs logged. `OLLAMA_BASE_URL` and
   `HF_SPACE_URL` can stay empty; Ollama is a local-dev backend and will never
   be reachable from Render.

5. **Apply**. First build takes 5–10 minutes — it is installing
   `backend/requirements.txt` from scratch.

6. Watch the deploy log for the startup banner. `backend/main.py`'s `lifespan`
   is loud on purpose and will tell you what is broken:

   - `Table creation failed:` → `DATABASE_URL` is wrong. Recheck the encoding
     and that you used the pooler host.
   - `Qdrant setup failed:` → `QDRANT_URL` / `QDRANT_API_KEY`.
   - `LLM_MODE=groq but GROQ_API_KEY is empty` → the secret did not save.
   - `HF_USERNAME is empty` → set it to `amijackofalltrades`.

7. Verify from your own machine:

   ```bash
   curl https://eka-backend.onrender.com/health
   ```

   You want `"status":"ok"` with `"database":true`, `"qdrant":true`,
   `"groq":true`. On the free tier expect `"complexity":{"tier":"heuristic"}`
   — that is intentional, see the RAM note below.

8. Run the real suite against it:

   ```bash
   EKA_BASE_URL=https://eka-backend.onrender.com python tests/test_e2e.py
   ```

---

## Route B — Manual web service

Only if Route A fails. You are hand-entering what `render.yaml` would have
done, so expect to consult that file for the full var list.

1. **render.com** → **New** → **Web Service**.
2. **Connect GitHub repo** → pick `eka` → **Connect**.
3. Fill in the service settings:

   | Field | Value |
   |---|---|
   | Name | `eka-backend` |
   | Region | `Singapore` (closest to India; Render has no Mumbai region) |
   | Branch | `main` |
   | Root Directory | `backend` |
   | Runtime | `Python 3` |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
   | Instance Type | `Free` |

4. **Advanced** → set **Health Check Path** to `/health`.
5. **Advanced** → **Add Environment Variable**, once per row. Copy the full
   list from `render.yaml`'s `envVars:` block — there are ~25. Do not skip
   `PYTHON_VERSION=3.11.9`; Render's default Python may not match what the
   pinned deps expect.
6. **Create Web Service**, then follow steps 6–8 from Route A.

---

## Free-tier realities

**512MB RAM, and torch does not fit.** `ENABLE_LOCAL_CLASSIFIERS=false` is not
a shortcut — `torch` + `transformers` is ~800MB on disk and 400–600MB resident
with a model loaded, which blows the box before FastAPI even starts.
`complexity_service.py` and `sentiment_service.py` fall back to
word-count/lexicon heuristics. The ranker (lightgbm, ~10MB, no torch) keeps
running in-process either way. To get the real fine-tuned classifiers you need
a Starter instance or the separate services in `serving/` — the commented-out
block at the bottom of `render.yaml` has both, with the cost tradeoff written
out.

**The instance sleeps after 15 minutes idle,** and a cold start is 50+
seconds. `render.yaml` sets `KEEP_ALIVE=1`, which starts the self-ping in
`main.py`. Understand its limit: **a self-ping cannot wake an instance that is
already suspended** — nothing inside a sleeping instance is running. It also
only counts as traffic when it goes out to `RENDER_EXTERNAL_URL` and back
through Render's router; the loopback fallback does nothing for spin-down.

So the self-ping is the backup, not the fix. Set up the external monitor in
`infra/uptimerobot_note.txt` — read it first, it covers the instance-hours
tradeoff (750 free hours/month is roughly one always-on service, so pinging
one service continuously consumes your whole month's allowance).

**Deploys are automatic** — `autoDeploy: true` means every push to `main`
redeploys. Turn it off in Settings if you want manual control.

---

## After it is up

- Frontend: point its `VITE_API_URL` (see `frontend/.env.example`) at the
  Render URL, then come back and set `CORS_ORIGINS` to that exact origin.
- API docs: `https://eka-backend.onrender.com/docs`.
- Rotate `SECRET_KEY` and any key that has ever been pasted into a chat
  window, a screenshot, or a terminal you did not control.
