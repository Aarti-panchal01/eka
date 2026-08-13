# Eka frontend

Single-page chat UI. Vite + React + Tailwind, no state library and no router —
there is one screen and it does not need either.

Everything reaches the backend through `src/api/ekaClient.js`. There are no
`fetch` calls in the UI, so the API surface stays in one file; if a route
changes, it changes there.

## Run it

```bash
cd frontend
npm install
cp .env.example .env          # set VITE_EKA_API_URL
npm run dev
```

`VITE_EKA_API_URL` defaults to `http://localhost:8000` when unset. **Vite
inlines it at build time**, so changing it means a rebuild and redeploy, not a
restart.

Start the backend separately:

```bash
cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8091
```

…and point `VITE_EKA_API_URL` at that port.

## What it does

- **Four modes** — founder / chanakya / gita / reflection. Switching modes
  starts a new session and clears the transcript, because the backend keys
  conversation history by session and mixing personas inside one is not a
  thing the pipeline expects.
- **Voice in** — hold the mic button. `MediaRecorder` → `POST /voice/stt` →
  the transcript is sent as a message.
- **Voice out** — the speaker toggle narrates replies via `POST /voice/tts`.
  Failures here surface as a banner and never block the text reply; voice is
  the first thing to 503 on a free tier.
- **Per-reply telemetry** — latency, which LLM backend answered, and any
  services that degraded on that request.

There is no auth. `getOrCreateUserId()` mints a browser-local id on first load
and persists it; every call carries it.

## Deploy to Vercel

`vercel.json` is set for a Vite SPA (build to `dist`, rewrite everything to
`/`). From this directory:

```bash
vercel --prod
```

Set **`VITE_EKA_API_URL`** in the Vercel project's environment variables to the
deployed backend URL before building, and set the backend's `CORS_ORIGINS` /
`FRONTEND_URL` to the Vercel domain, or the browser will block every request.
