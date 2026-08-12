# Migrating `Chat.jsx` from Base44 to the Eka backend

> **Note on scope:** this repo does not currently contain a
> `frontend/src/pages/Chat.jsx` — there's nothing to diff. What follows is a
> **pattern-based guide**: the shapes of Base44 SDK calls you're likely to have
> and the `ekaAPI` calls that replace them, plus complete, working code blocks
> you can drop in and adapt to whatever your actual component looks like. Treat
> the "before" column as illustrative of typical Base44 usage, not a
> transcription of your file.

## 1. Setup

1. **Place the client.** `ekaClient.js` lives at
   `frontend/src/api/ekaClient.js`. Import it as:
   ```js
   import ekaAPI, { getOrCreateUserId, EkaApiError } from "@/api/ekaClient";
   // or, without a path alias:
   import ekaAPI, { getOrCreateUserId, EkaApiError } from "../api/ekaClient";
   ```

2. **Environment.** Copy `frontend/.env.example` to `frontend/.env` and set
   `VITE_EKA_API_URL` to your backend (local `http://localhost:8000` while
   developing, your Render URL in production). Restart `vite dev` after
   creating/editing `.env` — and rebuild/redeploy for a production build, since
   Vite inlines the value at build time.

3. **Get a `user_id`.** The backend has no auth; every call needs a `user_id`
   the client generates and persists itself. Call this once, high in your
   component tree (or in a small context provider), and pass it down:
   ```js
   const userId = getOrCreateUserId(); // reads/writes localStorage["eka_user_id"]
   ```

## 2. Before / after: typical Base44 calls → ekaAPI

| Base44 SDK (typical)                              | ekaAPI equivalent                                              |
| --------------------------------------------------- | ---------------------------------------------------------------- |
| `base44.entities.Message.list({session_id})`        | `ekaAPI.getMessages(sessionId)`                                  |
| `base44.entities.Message.create({...})` + `base44.integrations.Core.InvokeLLM(...)` | `ekaAPI.sendMessage(sessionId, userId, message, mode)` — one call does both (persists the turn AND generates the reply) |
| `base44.entities.Conversation.list()`               | `ekaAPI.getSessions(userId, { includeArchived, limit })`         |
| `base44.entities.Conversation.create({...})`        | `ekaAPI.createSession(userId, mode, title)`                      |
| `base44.entities.Conversation.get(id)`              | `ekaAPI.getSession(sessionId)`                                   |
| `base44.entities.Conversation.update(id, {...})`    | `ekaAPI.updateSession(sessionId, updates)`                       |
| `base44.entities.Conversation.delete(id)`           | `ekaAPI.archiveSession(sessionId)` — soft delete, sets `archived: true`, does not hard-delete history |
| `base44.entities.Memory.list()`                     | `ekaAPI.getMemories(userId, filters)` — **returns `{items, total, skip, limit}`**, not a bare array |
| `base44.entities.Memory.create({...})`              | `ekaAPI.createMemory(memoryData)`                                |
| `base44.entities.Memory.update(id, {...})`          | `ekaAPI.updateMemory(memoryId, updates, userId)` — `userId` is a required query param, not part of `updates` |
| `base44.entities.Memory.delete(id)`                 | `ekaAPI.deleteMemory(memoryId, userId)` — same, `userId` required |
| `base44.integrations.Core.InvokeLLM({..., search: true})` | `ekaAPI.searchMemories(queryText, userId, limit)` — semantic search over stored memories |
| `base44.integrations.Core.UploadFile({file})`       | `ekaAPI.uploadFile(file, userId, { title, topic, importance })`  |
| `base44.integrations.Core.SpeechToText(blob)` (if used) | `ekaAPI.transcribe(audioBlob, filename)` -> `{text, backend}`  |
| `base44.integrations.Core.TextToSpeech(text)` (if used) | `ekaAPI.synthesize(text, mode)` -> `Blob` (audio/wav), then `ekaAPI.playAudio(blob)` |

## 3. Loading sessions on mount

```jsx
import { useEffect, useState } from "react";
import ekaAPI, { getOrCreateUserId, EkaApiError } from "../api/ekaClient";

function useSessions() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const userId = getOrCreateUserId();
    let cancelled = false;

    ekaAPI
      .getSessions(userId, { includeArchived: false, limit: 50 })
      .then((data) => {
        if (!cancelled) setSessions(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof EkaApiError ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { sessions, loading, error };
}
```

## 4. Creating a session

```jsx
async function startNewSession(mode = "founder", title = null) {
  const userId = getOrCreateUserId();
  const session = await ekaAPI.createSession(userId, mode, title);
  // session.id is the session_id you'll pass into sendMessage / getMessages
  return session;
}
```

## 5. Sending a message

### 5a. react-query / `useMutation` shape

```jsx
import { useMutation, useQueryClient } from "@tanstack/react-query";
import ekaAPI, { getOrCreateUserId } from "../api/ekaClient";

function useSendMessage(sessionId, setSessionId, mode) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (message) => {
      const userId = getOrCreateUserId();
      return ekaAPI.sendMessage(sessionId, userId, message, mode);
    },
    onSuccess: (response) => {
      // response.session_id is ALWAYS present — a brand-new chat gets one
      // back on its very first reply. Store it so the next turn continues
      // the same session instead of silently starting a new one each time.
      if (response.session_id && response.session_id !== sessionId) {
        setSessionId(response.session_id);
      }
      queryClient.invalidateQueries({ queryKey: ["messages", response.session_id] });
    },
  });
}

// usage in a component:
// const sendMessage = useSendMessage(sessionId, setSessionId, mode);
// sendMessage.mutate(input);
// sendMessage.isPending / sendMessage.data / sendMessage.error
```

### 5b. Plain `useState` shape (no react-query)

```jsx
import { useState } from "react";
import ekaAPI, { getOrCreateUserId, EkaApiError } from "../api/ekaClient";

function ChatPanel({ mode = "founder" }) {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [wakingUp, setWakingUp] = useState(false);

  async function handleSend() {
    if (!input.trim() || sending) return;
    const userId = getOrCreateUserId();
    const userText = input;
    setInput("");
    setSending(true);

    // Render's free tier spins down when idle; the first request after a
    // while can take 30-60s. Show a friendlier state after a short delay
    // instead of leaving a bare spinner up.
    const wakeTimer = setTimeout(() => setWakingUp(true), 4000);

    try {
      const response = await ekaAPI.sendMessage(sessionId, userId, userText, mode);
      if (response.session_id) setSessionId(response.session_id);
      setMessages((prev) => [
        ...prev,
        { role: "user", text: userText },
        { role: "eka", text: response.response, meta: response }, // see section 6
      ]);
    } catch (err) {
      const message = err instanceof EkaApiError ? err.message : String(err);
      setMessages((prev) => [...prev, { role: "system", text: `Error: ${message}` }]);
    } finally {
      clearTimeout(wakeTimer);
      setWakingUp(false);
      setSending(false);
    }
  }

  return (
    <div>
      {wakingUp && <p className="text-sm text-gray-400">Waking up Eka's backend, hang on…</p>}
      {/* render `messages`, an input bound to `input`/`setInput`, and a
          send button calling handleSend() */}
    </div>
  );
}
```

## 6. Rendering the response — the "Context used" indicator

`ChatResponse` includes `retrieved_memories: RetrievedMemory[]` and
`degraded: string[]`. Render a small "memories used" hint under each reply,
and optionally a dev-only badge for degraded services:

```jsx
function EkaMessage({ meta }) {
  const memCount = meta?.retrieved_memories?.length ?? 0;
  return (
    <div>
      <p>{meta.response}</p>
      {memCount > 0 && (
        <p style={{ fontSize: "0.75rem", color: "#888" }}>📎 Used {memCount} memories</p>
      )}
      {import.meta.env.DEV && meta?.degraded?.length > 0 && (
        <p style={{ fontSize: "0.7rem", color: "#c00" }}>
          degraded: {meta.degraded.join(", ")}
        </p>
      )}
    </div>
  );
}
```

`degraded` names which services fell back for this specific reply (e.g.
`["qdrant", "ranker"]`) — useful in dev to see when you're getting a
lower-quality answer path without digging through backend logs.

## 7. Voice-in flow (record → transcribe → auto-submit)

```jsx
import { useRef, useState } from "react";
import ekaAPI, { getOrCreateUserId } from "../api/ekaClient";

function useVoiceInput({ preferences, onTranscript, onAutoSubmit }) {
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const [recording, setRecording] = useState(false);

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];

    recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });

      try {
        const { text } = await ekaAPI.transcribe(blob, "voice-input.webm");
        onTranscript(text);
        // If the user has "always listening" on, skip the manual send button
        // and submit immediately.
        if (preferences?.always_listening && text.trim()) {
          onAutoSubmit(text);
        }
      } catch (err) {
        console.error("Transcription failed", err);
      }
    };

    mediaRecorderRef.current = recorder;
    recorder.start();
    setRecording(true);
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }

  return { recording, startRecording, stopRecording };
}
```

Wire `onTranscript` to `setInput`, and `onAutoSubmit` to the same
`handleSend`-style function from section 5b (called with the transcribed
text instead of waiting for a click).

## 8. Voice-out flow (response → synthesize → play)

```jsx
import ekaAPI, { EkaApiError } from "../api/ekaClient";

async function speakReply(responseText, mode, preferences) {
  if (!preferences?.voice_enabled) return;

  try {
    const audioBlob = await ekaAPI.synthesize(responseText, mode);
    const playback = ekaAPI.playAudio(audioBlob, {
      playbackSpeed: preferences.playback_speed ?? 1.0,
    });
    // playback is a Promise AND has .stop() — e.g. call playback.stop() if
    // the user starts typing a new message while Eka is still talking.
    await playback;
  } catch (err) {
    // 503 means TTS is unavailable (missing/expired SARVAM_API_KEY, no
    // credits, etc.) — this is expected to happen sometimes. Swallow it so
    // the text reply the user already sees is not treated as an error.
    if (!(err instanceof EkaApiError && err.status === 503)) {
      console.error("Voice playback failed", err);
    }
  }
}
```

Call `speakReply(response.response, mode, preferences)` right after a
successful `sendMessage` resolves.

## 9. Gotchas

- **Field is `response`, not `message`.** `ChatResponse.response` holds
  Eka's reply text. There is no `.message` field on the chat response.
- **`session_id` comes back on every reply.** Even on turn one with no
  session yet, the backend creates one and returns it — always capture and
  store `response.session_id`, don't assume you already have it from
  `createSession`.
- **`getMemories` returns an envelope.** `{items, total, skip, limit}` —
  read `.items`, don't map over the raw response.
- **`searchMemories` returns a bare array** (unlike `getMemories`) — no
  envelope there.
- **Deletes/updates on memory need `user_id` in the query string**, not the
  body: `deleteMemory(memoryId, userId)`, `updateMemory(memoryId, updates, userId)`,
  `updateMemoryPriority(memoryId, priority, userId)`. `updatePreferences`,
  by contrast, puts `user_id` in the JSON body — there's no per-resource
  path segment to attach a query param to there.
- **`archiveSession` takes no `user_id`** — it's a plain
  `DELETE /chat/sessions/{id}`.
- **CORS is already wide open** (`allow_origins=["*"]` in
  `backend/main.py`) — you do not need a dev proxy or CORS workaround
  locally or in production.
- **Render free tier cold starts.** If the backend has been idle, the first
  request can take 30-60 seconds while the instance spins back up. Don't
  use a short fetch timeout by default; if you add one, make it generous
  (60s+) for the first request of a session, and show a "waking up…" state
  rather than a bare spinner (see section 5b).
- **TTS 503 is expected, not exceptional.** Always catch it separately from
  other errors and just skip voice playback — never block the text reply on it.

## 10. Checklist

- [ ] `ekaClient.js` imported from `frontend/src/api/ekaClient.js`
- [ ] `.env` created from `.env.example` with `VITE_EKA_API_URL` set, dev
      server restarted (and prod rebuilt, if applicable)
- [ ] `getOrCreateUserId()` called once and threaded through the component
      instead of any Base44 auth/user object
- [ ] Session list loads via `getSessions`, new chats via `createSession`
- [ ] Sending uses `sendMessage` and persists `response.session_id`
- [ ] Message rendering reads `response.response` (not `.message`)
- [ ] "Context used" indicator wired to `retrieved_memories?.length`
- [ ] Voice-in wired: MediaRecorder → `transcribe` → input → optional
      auto-submit on `always_listening`
- [ ] Voice-out wired: `synthesize` → `playAudio`, 503 caught and ignored
- [ ] Memory screens use `.items` from `getMemories` and pass `userId` on
      every update/delete/priority call
- [ ] A "waking up" state exists for the first request after backend
      idle time
