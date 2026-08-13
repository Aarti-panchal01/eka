import { useCallback, useEffect, useRef, useState } from "react";

import ekaAPI, { EkaApiError, getOrCreateUserId } from "@/api/ekaClient";

/**
 * Single-page chat for Eka.
 *
 * Everything talks to the backend through ekaClient — no fetch calls live in
 * this file, so the API surface stays in one place. The backend has no auth;
 * `getOrCreateUserId()` mints and persists a browser-local id, and every call
 * carries it.
 */

const MODES = [
  { id: "founder", label: "Founder", hint: "Brutally honest operator", accent: "amber" },
  { id: "chanakya", label: "Chanakya", hint: "Strategy and leverage", accent: "violet" },
  { id: "gita", label: "Gita", hint: "Meaning when it hurts", accent: "sky" },
  { id: "reflection", label: "Reflection", hint: "Turns questions back", accent: "emerald" },
];

// Tailwind cannot see runtime-built class names, so the variants are spelled out.
const ACCENT = {
  amber: { on: "bg-amber-400 text-ink-900", off: "hover:border-amber-400/50 hover:text-amber-200", dot: "bg-amber-400" },
  violet: { on: "bg-violet-400 text-ink-900", off: "hover:border-violet-400/50 hover:text-violet-200", dot: "bg-violet-400" },
  sky: { on: "bg-sky-400 text-ink-900", off: "hover:border-sky-400/50 hover:text-sky-200", dot: "bg-sky-400" },
  emerald: { on: "bg-emerald-400 text-ink-900", off: "hover:border-emerald-400/50 hover:text-emerald-200", dot: "bg-emerald-400" },
};

const errText = (err) =>
  err instanceof EkaApiError
    ? `${err.message}${err.status ? ` (${err.status})` : ""}`
    : err?.message || "Something went wrong";

export default function App() {
  const [userId] = useState(() => getOrCreateUserId());
  const [mode, setMode] = useState("founder");
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [health, setHealth] = useState("checking");
  const [banner, setBanner] = useState(null);

  // voice
  const [speakReplies, setSpeakReplies] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const playbackRef = useRef(null);

  const transcriptRef = useRef(null);
  const inputRef = useRef(null);

  const accent = ACCENT[MODES.find((m) => m.id === mode).accent];

  // ---------------------------------------------------------------- health
  useEffect(() => {
    let alive = true;
    ekaAPI
      .checkHealth()
      .then(() => alive && setHealth("up"))
      .catch(() => alive && setHealth("down"));
    return () => {
      alive = false;
    };
  }, []);

  // Keep the newest message in view without yanking the page around.
  useEffect(() => {
    const el = transcriptRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, sending]);

  // Stop any in-flight narration when the component goes away.
  useEffect(() => () => playbackRef.current?.stop?.(), []);

  // ------------------------------------------------------------------ send
  const send = useCallback(
    async (text) => {
      const body = text.trim();
      if (!body || sending) return;

      setBanner(null);
      setDraft("");
      setMessages((prev) => [...prev, { role: "user", content: body }]);
      setSending(true);

      try {
        const res = await ekaAPI.sendMessage(sessionId, userId, body, mode);
        // Every reply carries the current session id — store it back, because
        // the first send is made with null and the backend creates the session.
        if (res.session_id) setSessionId(res.session_id);

        setMessages((prev) => [
          ...prev,
          {
            role: "eka",
            content: res.response,
            mode: res.mode,
            latency: res.latency_ms,
            backend: res.llm_backend,
            degraded: res.degraded || [],
          },
        ]);

        if (speakReplies) narrate(res.response, res.mode || mode);
      } catch (err) {
        setBanner(errText(err));
        setMessages((prev) => [
          ...prev,
          { role: "system", content: `Could not send: ${errText(err)}` },
        ]);
      } finally {
        setSending(false);
        inputRef.current?.focus();
      }
    },
    [mode, sending, sessionId, speakReplies, userId]
  );

  // ----------------------------------------------------------------- voice
  async function narrate(text, forMode) {
    try {
      playbackRef.current?.stop?.();
      const blob = await ekaAPI.synthesize(text, forMode);
      playbackRef.current = ekaAPI.playAudio(blob);
      await playbackRef.current;
    } catch (err) {
      // Voice is a bonus, never a blocker — a 503 here must not eat the reply.
      setBanner(`Voice output unavailable — ${errText(err)}`);
    }
  }

  async function startRecording() {
    setBanner(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setBanner("This browser has no microphone API.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        if (!blob.size) return;

        setTranscribing(true);
        try {
          const { text } = await ekaAPI.transcribe(blob);
          if (text?.trim()) send(text);
          else setBanner("Nothing was picked up — try again.");
        } catch (err) {
          setBanner(`Transcription failed — ${errText(err)}`);
        } finally {
          setTranscribing(false);
        }
      };

      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
    } catch {
      setBanner("Microphone permission denied.");
    }
  }

  function stopRecording() {
    recorderRef.current?.state === "recording" && recorderRef.current.stop();
    setRecording(false);
  }

  function switchMode(next) {
    if (next === mode) return;
    playbackRef.current?.stop?.();
    setMode(next);
    // A new persona is a new conversation; the backend keys history by session.
    setSessionId(null);
    setMessages([]);
    setBanner(null);
  }

  const busy = sending || transcribing;

  return (
    <div className="flex h-full flex-col">
      {/* ------------------------------------------------------------ head */}
      <header className="border-b border-ink-700 px-4 py-3 sm:px-6">
        <div className="mx-auto flex max-w-3xl items-center gap-3">
          <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${accent.dot}`} />
          <h1 className="text-lg font-semibold tracking-tight">Eka</h1>
          <span
            className="ml-auto flex items-center gap-1.5 text-xs text-neutral-400"
            title={health === "up" ? "Backend reachable" : "Backend unreachable"}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                health === "up"
                  ? "bg-emerald-400"
                  : health === "down"
                    ? "bg-red-400"
                    : "bg-neutral-500"
              }`}
            />
            {health === "up" ? "connected" : health === "down" ? "offline" : "…"}
          </span>
        </div>

        <div className="mx-auto mt-3 flex max-w-3xl flex-wrap gap-2">
          {MODES.map((m) => {
            const a = ACCENT[m.accent];
            const active = m.id === mode;
            return (
              <button
                key={m.id}
                onClick={() => switchMode(m.id)}
                title={m.hint}
                className={`rounded-full border px-3 py-1.5 text-sm font-medium transition ${
                  active
                    ? `border-transparent ${a.on}`
                    : `border-ink-600 text-neutral-300 ${a.off}`
                }`}
              >
                {m.label}
              </button>
            );
          })}
        </div>
      </header>

      {/* ------------------------------------------------------ transcript */}
      <main ref={transcriptRef} className="scroll-thin flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.length === 0 && (
            <div className="mt-10 text-center text-neutral-400">
              <p className="text-base">
                {MODES.find((m) => m.id === mode).hint}.
              </p>
              <p className="mt-1 text-sm text-neutral-500">
                Type below, or hold the mic to speak.
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <Bubble key={i} msg={msg} accent={accent} />
          ))}

          {sending && (
            <div className="animate-rise self-start rounded-2xl bg-ink-800 px-4 py-3 text-neutral-400">
              <span className="inline-flex gap-1">
                <Dot /> <Dot delay="150ms" /> <Dot delay="300ms" />
              </span>
            </div>
          )}
        </div>
      </main>

      {/* ---------------------------------------------------------- footer */}
      <footer className="border-t border-ink-700 px-4 py-3 sm:px-6">
        <div className="mx-auto max-w-3xl">
          {banner && (
            <p className="mb-2 rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-300">
              {banner}
            </p>
          )}

          <form
            onSubmit={(e) => {
              e.preventDefault();
              send(draft);
            }}
            className="flex items-end gap-2"
          >
            <button
              type="button"
              onClick={() => setSpeakReplies((v) => !v)}
              title={speakReplies ? "Spoken replies on" : "Spoken replies off"}
              aria-pressed={speakReplies}
              className={`shrink-0 rounded-xl border px-3 py-2.5 text-sm transition ${
                speakReplies
                  ? "border-transparent bg-ink-600 text-neutral-100"
                  : "border-ink-600 text-neutral-400 hover:text-neutral-200"
              }`}
            >
              {speakReplies ? "🔊" : "🔇"}
            </button>

            <textarea
              ref={inputRef}
              rows={1}
              value={draft}
              disabled={busy}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                // Enter sends; Shift+Enter is a newline.
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(draft);
                }
              }}
              placeholder={transcribing ? "Transcribing…" : "Say what's actually going on…"}
              className="max-h-40 min-h-[46px] flex-1 resize-y rounded-xl border border-ink-600 bg-ink-800 px-3.5 py-2.5 text-[15px] placeholder:text-neutral-500 focus:border-neutral-500 focus:outline-none disabled:opacity-60"
            />

            <button
              type="button"
              onMouseDown={startRecording}
              onMouseUp={stopRecording}
              onMouseLeave={() => recording && stopRecording()}
              onTouchStart={(e) => {
                e.preventDefault();
                startRecording();
              }}
              onTouchEnd={(e) => {
                e.preventDefault();
                stopRecording();
              }}
              disabled={transcribing}
              title="Hold to talk"
              className={`shrink-0 rounded-xl border px-3 py-2.5 text-sm transition ${
                recording
                  ? "animate-pulseRing border-transparent bg-red-500 text-white"
                  : "border-ink-600 text-neutral-400 hover:text-neutral-200"
              } disabled:opacity-50`}
            >
              🎙
            </button>

            <button
              type="submit"
              disabled={busy || !draft.trim()}
              className={`shrink-0 rounded-xl px-4 py-2.5 text-sm font-semibold transition ${accent.on} disabled:cursor-not-allowed disabled:opacity-40`}
            >
              Send
            </button>
          </form>
        </div>
      </footer>
    </div>
  );
}

function Bubble({ msg, accent }) {
  if (msg.role === "system") {
    return (
      <div className="animate-rise self-center rounded-lg bg-red-500/10 px-3 py-1.5 text-xs text-red-300">
        {msg.content}
      </div>
    );
  }

  const mine = msg.role === "user";
  return (
    <div className={`flex ${mine ? "justify-end" : "justify-start"}`}>
      <div
        className={`animate-rise max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-[15px] leading-relaxed ${
          mine ? `${accent.on} rounded-br-md` : "rounded-bl-md bg-ink-800 text-neutral-100"
        }`}
      >
        {msg.content}
        {!mine && (msg.latency != null || msg.degraded?.length > 0) && (
          <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-neutral-500">
            {msg.latency != null && <span>{msg.latency} ms</span>}
            {msg.backend && <span>· {msg.backend}</span>}
            {msg.degraded?.length > 0 && (
              <span className="text-amber-400/80">· degraded: {msg.degraded.join(", ")}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const Dot = ({ delay = "0ms" }) => (
  <span
    className="inline-block h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-500"
    style={{ animationDelay: delay }}
  />
);
