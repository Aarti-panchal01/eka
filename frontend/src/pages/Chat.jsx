import { useCallback, useEffect, useRef, useState } from "react";

import ekaAPI from "@/api/ekaClient";
import { errText, modeOf, userId } from "@/lib/ui";

export default function Chat({ mode, health }) {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [waking, setWaking] = useState(false);
  const [toast, setToast] = useState(null);

  const [emotion, setEmotion] = useState(false);
  const [speak, setSpeak] = useState(false);
  const [listening, setListening] = useState(false);

  const recognition = useRef(null);
  const playback = useRef(null);
  const transcript = useRef(null);
  const input = useRef(null);

  const persona = modeOf(mode);

  // Toasts auto-clear; a mic error should not sit on screen forever.
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 5000);
    return () => clearTimeout(t);
  }, [toast]);

  // A new persona is a new conversation — the backend keys history by session.
  useEffect(() => {
    playback.current?.stop?.();
    recognition.current?.abort?.();
    setSessionId(null);
    setMessages([]);
    setToast(null);
  }, [mode]);

  useEffect(() => {
    const el = transcript.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, sending]);

  useEffect(
    () => () => {
      playback.current?.stop?.();
      recognition.current?.abort?.();
    },
    []
  );

  const narrate = useCallback(async (text, forMode) => {
    try {
      playback.current?.stop?.();
      const blob = await ekaAPI.synthesize(text, forMode);
      playback.current = ekaAPI.playAudio(blob);
      await playback.current;
    } catch {
      // Voice is a bonus. A 503 here must never touch the text reply.
      setToast("Voice output unavailable right now.");
    }
  }, []);

  const send = useCallback(
    async (text) => {
      const body = text.trim();
      if (!body || sending) return;

      setToast(null);
      setDraft("");
      setMessages((p) => [...p, { role: "user", content: body }]);
      setSending(true);
      const wake = setTimeout(() => setWaking(true), 4000);

      try {
        const res = await ekaAPI.sendMessage(sessionId, userId(), body, mode);
        // Turn one goes out with null and the backend mints the session. Store
        // it back or every message silently starts a new conversation.
        if (res.session_id) setSessionId(res.session_id);
        setMessages((p) => [
          ...p,
          {
            role: "eka",
            content: res.response,
            mode: res.mode || mode,
            memories: res.retrieved_memories?.length ?? 0,
          },
        ]);
        if (speak) narrate(res.response, res.mode || mode);
      } catch (err) {
        setToast(errText(err));
        setMessages((p) => [
          ...p,
          { role: "system", content: `Could not send: ${errText(err)}` },
        ]);
      } finally {
        clearTimeout(wake);
        setWaking(false);
        setSending(false);
        input.current?.focus();
      }
    },
    [mode, narrate, sending, sessionId, speak]
  );

  /**
   * Speech-to-text runs entirely in the browser.
   *
   * The Web Speech API needs no backend call, so there is no CORS surface and
   * no round trip — the transcript lands in the input box in about a second.
   * It is Chromium-only, which is why the unsupported branch says so plainly
   * rather than failing silently on Firefox.
   */
  function toggleMic() {
    if (listening) {
      recognition.current?.stop();
      return;
    }
    const SpeechRec =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      setToast("Mic not supported in this browser — try Chrome or Edge.");
      return;
    }

    const rec = new SpeechRec();
    rec.lang = "en-IN";
    rec.continuous = false;
    rec.interimResults = false;

    rec.onresult = (e) => {
      const text = e.results[0][0].transcript;
      setDraft((prev) => (prev ? `${prev} ${text}` : text));
      input.current?.focus();
    };
    rec.onerror = (e) => {
      setListening(false);
      setToast(
        e.error === "not-allowed"
          ? "Microphone permission denied."
          : `Mic error: ${e.error}`
      );
    };
    rec.onend = () => setListening(false);

    recognition.current = rec;
    setListening(true);
    rec.start();
  }

  return (
    <div className="flex h-full flex-col bg-ink">
      <header className="flex items-center gap-3 border-b border-edge px-7 py-4">
        <span className="text-xl">{persona.icon}</span>
        <div className="min-w-0">
          <h1 className={`text-base font-semibold leading-none ${persona.accent}`}>
            {persona.label}
          </h1>
          <p className="mt-1 text-xs text-neutral-500">{persona.hint}</p>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setEmotion((v) => !v)}
            title="Adapt tone to how you sound"
            className={`chip transition-all duration-200 ${
              emotion ? "border-gold/60 text-gold" : "hover:text-neutral-200"
            }`}
          >
            ♥ Emotion
          </button>
          <button
            onClick={() => setSpeak((v) => !v)}
            title="Read replies aloud"
            className={`chip transition-all duration-200 ${
              speak ? "border-gold/60 text-gold" : "hover:text-neutral-200"
            }`}
          >
            {speak ? "🔊 Voice" : "🔇 Voice"}
          </button>
          <span
            title={health === "up" ? "Connected" : "Backend unreachable"}
            className={`ml-1 h-2 w-2 rounded-full ${
              health === "up"
                ? "animate-pulse bg-emerald-400"
                : health === "down"
                  ? "bg-red-400"
                  : "bg-neutral-600"
            }`}
          />
        </div>
      </header>

      <main ref={transcript} className="scroll-thin flex-1 overflow-y-auto px-7 py-7">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.length === 0 && (
            <div className="mt-20 text-center">
              <p className="text-4xl">{persona.icon}</p>
              <p className={`mt-4 text-lg ${persona.accent}`}>{persona.hint}.</p>
              <p className="mt-1 text-sm text-neutral-600">
                Type below, or tap the mic to speak.
              </p>
            </div>
          )}

          {messages.map((m, i) => (
            <Bubble key={i} msg={m} />
          ))}

          {sending && (
            <div className="card animate-rise self-start px-4 py-3 text-neutral-500 shadow-card">
              {waking ? (
                <span className="text-sm">Waking Eka up, hang on…</span>
              ) : (
                <span className="inline-flex gap-1">
                  <Dot /> <Dot d="150ms" /> <Dot d="300ms" />
                </span>
              )}
            </div>
          )}
        </div>
      </main>

      <footer className="border-t border-edge px-7 py-4">
        <div className="mx-auto max-w-3xl">
          {toast && (
            <p className="mb-2 rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-300">
              {toast}
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
              disabled={sending}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(draft);
                }
              }}
              placeholder="Say what's actually going on..."
              className="field max-h-40 min-h-[48px] flex-1 resize-y"
            />
            <button
              type="button"
              onClick={toggleMic}
              title={listening ? "Stop listening" : "Tap to talk"}
              className={`shrink-0 rounded-xl border px-4 py-3 text-lg transition-all duration-200 ${
                listening
                  ? "animate-pulseRing border-transparent bg-red-500 text-white"
                  : "border-edge text-neutral-400 hover:border-neutral-600 hover:text-neutral-100"
              }`}
            >
              🎙
            </button>
            <button
              type="submit"
              disabled={sending || !draft.trim()}
              className="btn-gold shrink-0 !px-5 !py-3"
            >
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

  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="animate-rise max-w-[70%] rounded-[18px] bg-gold px-4 py-3 text-[15px] font-medium leading-relaxed text-ink shadow-[0_2px_16px_-6px_#f5a623]">
          {msg.content}
        </div>
      </div>
    );
  }

  const persona = modeOf(msg.mode);
  return (
    <div className="animate-rise w-full">
      <div className="card px-5 py-4 text-[15px] leading-relaxed text-neutral-100 shadow-card">
        <p className="whitespace-pre-wrap">{msg.content}</p>
      </div>
      <div className="mt-1.5 flex items-center gap-2 pl-1 text-[11px]">
        <span className={persona.accent}>{persona.id} mode</span>
        {msg.memories > 0 && (
          <span className="text-neutral-600">· {msg.memories} memories used</span>
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
