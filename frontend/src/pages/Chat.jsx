import { useCallback, useEffect, useRef, useState } from "react";

import ekaAPI from "@/api/ekaClient";
import { MODES, errText, userId } from "@/lib/ui";

export default function Chat({ mode }) {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [waking, setWaking] = useState(false);
  const [banner, setBanner] = useState(null);

  // voice
  const [emotion, setEmotion] = useState(false);
  const [speak, setSpeak] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const recorder = useRef(null);
  const chunks = useRef([]);
  const playback = useRef(null);

  const transcript = useRef(null);
  const input = useRef(null);
  const persona = MODES.find((m) => m.id === mode);

  // A new persona is a new conversation: the backend keys history by session
  // and the pipeline does not expect two personas inside one.
  useEffect(() => {
    playback.current?.stop?.();
    setSessionId(null);
    setMessages([]);
    setBanner(null);
  }, [mode]);

  useEffect(() => {
    const el = transcript.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, sending]);

  useEffect(() => () => playback.current?.stop?.(), []);

  const narrate = useCallback(async (text, forMode) => {
    try {
      playback.current?.stop?.();
      const blob = await ekaAPI.synthesize(text, forMode);
      playback.current = ekaAPI.playAudio(blob);
      await playback.current;
    } catch (err) {
      // TTS is the first thing to 503 on a free tier. It must never take the
      // text reply down with it.
      setBanner(`Voice unavailable — ${errText(err)}`);
    }
  }, []);

  const send = useCallback(
    async (text) => {
      const body = text.trim();
      if (!body || sending) return;

      setBanner(null);
      setDraft("");
      setMessages((p) => [...p, { role: "user", content: body }]);
      setSending(true);
      const wakeTimer = setTimeout(() => setWaking(true), 4000);

      try {
        const res = await ekaAPI.sendMessage(sessionId, userId(), body, mode);
        // Always store this back: turn one goes out with null and the backend
        // mints the session. Miss it and every message starts a new chat.
        if (res.session_id) setSessionId(res.session_id);
        setMessages((p) => [
          ...p,
          {
            role: "eka",
            content: res.response,
            memories: res.retrieved_memories?.length ?? 0,
            latency: res.latency_ms,
            backend: res.llm_backend,
            sentiment: res.sentiment,
            degraded: res.degraded || [],
          },
        ]);
        if (speak) narrate(res.response, res.mode || mode);
      } catch (err) {
        setBanner(errText(err));
        setMessages((p) => [
          ...p,
          { role: "system", content: `Could not send: ${errText(err)}` },
        ]);
      } finally {
        clearTimeout(wakeTimer);
        setWaking(false);
        setSending(false);
        input.current?.focus();
      }
    },
    [mode, narrate, sending, sessionId, speak]
  );

  async function startRec() {
    setBanner(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setBanner("This browser has no microphone API.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      chunks.current = [];
      rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks.current, { type: "audio/webm" });
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
      rec.start();
      recorder.current = rec;
      setRecording(true);
    } catch {
      setBanner("Microphone permission denied.");
    }
  }

  function stopRec() {
    if (recorder.current?.state === "recording") recorder.current.stop();
    setRecording(false);
  }

  const busy = sending || transcribing;

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-3 border-b border-edge px-6 py-4">
        <span className="text-gold">{persona.glyph}</span>
        <div>
          <h1 className="text-base font-semibold leading-none">{persona.label}</h1>
          <p className="mt-1 text-xs text-neutral-500">{persona.hint}</p>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setEmotion((v) => !v)}
            title="Emotion mode — reads sentiment and adapts tone"
            className={`chip transition ${
              emotion ? "border-gold/50 text-gold" : "hover:text-neutral-200"
            }`}
          >
            ♥ Emotion {emotion ? "on" : "off"}
          </button>
          <button
            onClick={() => setSpeak((v) => !v)}
            title={speak ? "Spoken replies on" : "Spoken replies off"}
            className={`chip transition ${
              speak ? "border-gold/50 text-gold" : "hover:text-neutral-200"
            }`}
          >
            {speak ? "🔊 Voice on" : "🔇 Voice off"}
          </button>
        </div>
      </header>

      <main ref={transcript} className="scroll-thin flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.length === 0 && (
            <div className="mt-16 text-center">
              <p className="text-3xl text-gold">{persona.glyph}</p>
              <p className="mt-3 text-neutral-300">{persona.hint}.</p>
              <p className="mt-1 text-sm text-neutral-600">
                Type below, or hold the mic to speak.
              </p>
            </div>
          )}

          {messages.map((m, i) => (
            <Bubble key={i} msg={m} />
          ))}

          {sending && (
            <div className="card animate-rise self-start px-4 py-3 text-neutral-500">
              {waking ? (
                <span className="text-sm">Waking Eka's backend, hang on…</span>
              ) : (
                <span className="inline-flex gap-1">
                  <Dot /> <Dot d="150ms" /> <Dot d="300ms" />
                </span>
              )}
            </div>
          )}
        </div>
      </main>

      <footer className="border-t border-edge px-6 py-4">
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
            <textarea
              ref={input}
              rows={1}
              value={draft}
              disabled={busy}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(draft);
                }
              }}
              placeholder={
                transcribing ? "Transcribing…" : "Say what's actually going on…"
              }
              className="field max-h-40 min-h-[46px] flex-1 resize-y"
            />
            <button
              type="button"
              onMouseDown={startRec}
              onMouseUp={stopRec}
              onMouseLeave={() => recording && stopRec()}
              onTouchStart={(e) => {
                e.preventDefault();
                startRec();
              }}
              onTouchEnd={(e) => {
                e.preventDefault();
                stopRec();
              }}
              disabled={transcribing}
              title="Hold to talk"
              className={`shrink-0 rounded-lg border px-3.5 py-2.5 transition ${
                recording
                  ? "animate-pulseRing border-transparent bg-red-500 text-white"
                  : "border-edge text-neutral-400 hover:text-neutral-100"
              } disabled:opacity-40`}
            >
              🎙
            </button>
            <button type="submit" disabled={busy || !draft.trim()} className="btn-gold shrink-0">
              Send
            </button>
          </form>
        </div>
      </footer>
    </div>
  );
}

function Bubble({ msg }) {
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
          mine
            ? "rounded-br-md bg-gold font-medium text-ink"
            : "card rounded-bl-md text-neutral-100"
        }`}
      >
        {msg.content}
        {!mine && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-neutral-600">
            {/* Surfacing retrieval makes the memory pipeline visible instead of
                magic — and makes a bad recall obvious. */}
            {msg.memories > 0 && (
              <span className="text-gold/70">📎 {msg.memories} memories</span>
            )}
            {msg.latency != null && <span>{msg.latency} ms</span>}
            {msg.backend && <span>· {msg.backend}</span>}
            {msg.degraded?.length > 0 && (
              <span className="text-amber-400/80">
                · degraded: {msg.degraded.join(", ")}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const Dot = ({ d = "0ms" }) => (
  <span
    className="inline-block h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-600"
    style={{ animationDelay: d }}
  />
);
