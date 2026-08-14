import { useCallback, useEffect, useRef, useState } from "react";

import ekaAPI from "@/api/ekaClient";
import {
  errText,
  getSession,
  langOf,
  modeOf,
  setSession,
  userId,
} from "@/lib/ui";

export default function Chat({ mode, language, health, newChatToken }) {
  const [sessionId, setSessionId] = useState(() => getSession(mode));
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [waking, setWaking] = useState(false);
  const [toast, setToast] = useState(null);
  const [resumed, setResumed] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const [emotion, setEmotion] = useState(false);
  const [speak, setSpeak] = useState(false);
  const [listening, setListening] = useState(false);

  const recognition = useRef(null);
  const playback = useRef(null);
  const transcript = useRef(null);
  const input = useRef(null);

  const persona = modeOf(mode);
  const lang = langOf(language);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 5000);
    return () => clearTimeout(t);
  }, [toast]);

  /**
   * Switching mode resumes that mode's own conversation.
   *
   * The backend takes `mode` per message rather than per session, so a session
   * survives a mode switch — there is no reason to throw the history away, and
   * doing so was what made Eka feel amnesiac.
   */
  useEffect(() => {
    playback.current?.stop?.();
    recognition.current?.abort?.();
    setToast(null);

    const saved = getSession(mode);
    setSessionId(saved);
    setMessages([]);
    setResumed(false);

    if (!saved) return;
    let alive = true;
    setLoadingHistory(true);
    ekaAPI
      .getMessages(saved, 200)
      .then((rows) => {
        if (!alive) return;
        const history = (Array.isArray(rows) ? rows : (rows?.items ?? [])).map(
          (m) => ({
            role: m.role === "user" ? "user" : "eka",
            content: m.content ?? m.text ?? "",
            mode: m.mode || mode,
          })
        );
        setMessages(history);
        setResumed(history.length > 0);
      })
      .catch(() => {
        // A stale session id (server-side cleanup, different device) should not
        // block a new conversation — drop it and start fresh.
        if (alive) setSession(mode, null);
      })
      .finally(() => alive && setLoadingHistory(false));
    return () => {
      alive = false;
    };
  }, [mode]);

  // "New chat" bumps a token in App rather than calling in here.
  useEffect(() => {
    if (!newChatToken) return;
    playback.current?.stop?.();
    setSession(mode, null);
    setSessionId(null);
    setMessages([]);
    setResumed(false);
    setToast(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [newChatToken]);

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

  const narrate = useCallback(
    async (text, forMode) => {
      try {
        playback.current?.stop?.();
        const blob = await ekaAPI.synthesize(text, forMode, language);
        playback.current = ekaAPI.playAudio(blob);
        await playback.current;
      } catch {
        setToast("Voice output unavailable right now.");
      }
    },
    [language]
  );

  const send = useCallback(
    async (text) => {
      const body = text.trim();
      if (!body || sending) return;

      setToast(null);
      setResumed(false);
      setDraft("");
      setMessages((p) => [...p, { role: "user", content: body }]);
      setSending(true);
      const wake = setTimeout(() => setWaking(true), 4000);

      try {
        const res = await ekaAPI.sendMessage(
          sessionId,
          userId(),
          body,
          mode,
          language
        );
        if (res.session_id && res.session_id !== sessionId) {
          setSessionId(res.session_id);
          setSession(mode, res.session_id);
        }
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
    [language, mode, narrate, sending, sessionId, speak]
  );

  /**
   * Speech-to-text runs entirely in the browser: no backend call, no CORS
   * surface, transcript in about a second. Chromium-only, so the unsupported
   * branch says so rather than failing silently on Firefox.
   */
  function toggleMic() {
    if (listening) {
      recognition.current?.stop();
      return;
    }
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      setToast("Mic not supported in this browser — try Chrome or Edge.");
      return;
    }
    const rec = new SpeechRec();
    rec.lang = language; // en-IN / hi-IN / kn-IN
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
        <span className="chip ml-1" title={lang.label}>
          {lang.flag} {lang.short}
        </span>

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
          {resumed && (
            <div className="self-center rounded-full border border-edge bg-card px-3 py-1 text-[11px] text-neutral-500">
              ↩ continuing your {persona.label.toLowerCase()} conversation
            </div>
          )}

          {loadingHistory && messages.length === 0 && (
            <p className="mt-20 text-center text-sm text-neutral-600">
              Loading your conversation…
            </p>
          )}

          {!loadingHistory && messages.length === 0 && (
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
            <div className="card animate-rise self-start px-4 py-3 text-neutral-500">
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
      <div className="card px-5 py-4 text-[15px] leading-relaxed text-neutral-100">
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
