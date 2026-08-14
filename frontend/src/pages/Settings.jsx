import { useEffect, useState } from "react";

import ekaAPI from "@/api/ekaClient";
import {
  ErrorBox,
  LANGUAGES,
  Loading,
  MODES,
  PageHeader,
  Row,
  Toggle,
  errText,
  fmtDate,
  useAsync,
  userId,
} from "@/lib/ui";

// Our labels; the ids are what the backend's voice config expects.
const VOICES = [
  { id: "male_deep", label: "Male · Deep" },
  { id: "male_warm", label: "Male · Warm" },
  { id: "female_serene", label: "Female · Serene" },
  { id: "alien_ethereal", label: "Alien · Ethereal" },
];

export default function Settings({ mode, onMode, language, onLanguage }) {
  const { data, loading, error, reload } = useAsync(
    () => ekaAPI.getPreferences(userId()),
    []
  );
  // Counts for "About your Eka" — failures here must not break the page, so
  // each falls back to null and renders as "—".
  const { data: stats } = useAsync(
    () =>
      Promise.all([
        ekaAPI.getMemories(userId(), { limit: 1 }).catch(() => null),
        ekaAPI.getSessions(userId(), { limit: 200 }).catch(() => null),
      ]).then(([mem, sessions]) => ({
        memories: mem?.total ?? null,
        sessions: Array.isArray(sessions) ? sessions.length : null,
        since:
          Array.isArray(sessions) && sessions.length
            ? sessions[sessions.length - 1]?.created_date
            : null,
      })),
    []
  );

  const [prefs, setPrefs] = useState({});
  const [saving, setSaving] = useState(false);
  const [note, setNote] = useState(null);
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (data) setPrefs(data);
  }, [data]);

  async function save(patch) {
    setPrefs((p) => ({ ...p, ...patch }));
    setSaving(true);
    setNote(null);
    try {
      await ekaAPI.updatePreferences(userId(), patch);
    } catch (err) {
      setNote(errText(err));
      reload();
    } finally {
      setSaving(false);
    }
  }

  function copyId() {
    navigator.clipboard?.writeText(userId());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  /**
   * Page through every memory.
   *
   * The route caps `limit` at 100 (422 above that), so "just ask for 500" is
   * not available — and silently taking the first 100 would make an export
   * that looks complete and is not. Stops on a short page or when `total` is
   * reached, with a hard page cap so a bad `total` cannot loop forever.
   */
  async function fetchAllMemories() {
    const PAGE = 100;
    const out = [];
    for (let page = 0; page < 100; page += 1) {
      const res = await ekaAPI.getMemories(userId(), {
        skip: page * PAGE,
        limit: PAGE,
      });
      const items = res?.items ?? [];
      out.push(...items);
      if (items.length < PAGE) break;
      if (res?.total != null && out.length >= res.total) break;
    }
    return out;
  }

  async function exportMemories() {
    setBusy(true);
    setNote(null);
    try {
      const items = await fetchAllMemories();
      if (!items.length) {
        setNote("Nothing to export yet.");
        return;
      }
      const blob = new Blob([JSON.stringify(items, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `eka-memories-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setNote(errText(err));
    } finally {
      setBusy(false);
    }
  }

  async function clearMemories() {
    // Irreversible and server-side, so it gets a real confirmation rather
    // than a toast-and-undo that this API cannot honour.
    if (
      !window.confirm(
        "Delete every memory Eka has of you? This cannot be undone."
      )
    )
      return;
    setBusy(true);
    setNote(null);
    try {
      const items = await fetchAllMemories();
      for (const m of items) {
        await ekaAPI.deleteMemory(m.id, userId());
      }
      setNote(`Deleted ${items.length} memories.`);
    } catch (err) {
      setNote(errText(err));
    } finally {
      setBusy(false);
    }
  }

  function clearHistory() {
    if (!window.confirm("Start fresh conversations in all four modes?")) return;
    // Session ids are browser-local. Clearing them starts new conversations
    // without deleting anything server-side.
    for (const m of MODES) localStorage.removeItem(`eka_session_${m.id}`);
    setNote("Conversation history cleared. Reload to see empty chats.");
  }

  if (loading)
    return (
      <Shell>
        <Loading label="Loading preferences…" />
      </Shell>
    );
  if (error)
    return (
      <Shell>
        <ErrorBox onRetry={reload}>{error}</ErrorBox>
      </Shell>
    );

  return (
    <Shell>
      <PageHeader
        title="Settings"
        subtitle={saving ? "Saving…" : "Changes save as you make them."}
      />
      {note && <ErrorBox>{note}</ErrorBox>}

      <H>About your Eka</H>
      <section className="card mb-7 px-5">
        <Row label="User ID" hint="Your entire identity. There is no login">
          <button onClick={copyId} className="btn-ghost font-mono !text-xs">
            {copied ? "copied ✓" : `${userId().slice(0, 8)}… copy`}
          </button>
        </Row>
        <Row label="Memories saved">
          <span className="text-sm tabular-nums text-gold">
            {stats?.memories ?? "-"}
          </span>
        </Row>
        <Row label="Conversations">
          <span className="text-sm tabular-nums text-gold">
            {stats?.sessions ?? "-"}
          </span>
        </Row>
        <Row label="Member since">
          <span className="text-sm text-neutral-400">
            {stats?.since ? fmtDate(stats.since) : "-"}
          </span>
        </Row>
      </section>

      <H>Persona</H>
      <div className="mb-4 grid grid-cols-2 gap-3">
        {MODES.map((m) => {
          const active = m.id === mode;
          return (
            <button
              key={m.id}
              onClick={() => onMode(m.id)}
              className={`rounded-xl border px-4 py-3.5 text-left transition-all duration-200 ${
                active ? m.selected : `border-edge bg-card ${m.idle}`
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="text-base">{m.icon}</span>
                <span
                  className={`text-sm font-semibold ${
                    active ? m.selectedText : "text-neutral-200"
                  }`}
                >
                  {m.label}
                </span>
              </div>
              <p
                className={`mt-1 pl-7 text-[11px] ${
                  active ? m.selectedHint : "text-neutral-500"
                }`}
              >
                {m.hint}
              </p>
            </button>
          );
        })}
      </div>

      <div className="mb-4 grid grid-cols-3 gap-3">
        {LANGUAGES.map((l) => {
          const active = l.id === language;
          return (
            <button
              key={l.id}
              onClick={() => onLanguage(l.id)}
              className={`rounded-xl border px-4 py-3 text-center transition-all duration-200 ${
                active
                  ? "border-gold bg-gold-soft text-gold"
                  : "border-edge bg-card text-neutral-300 hover:border-gold/40"
              }`}
            >
              <div className="text-xl">{l.flag}</div>
              <div className="mt-1 text-sm font-medium">{l.label}</div>
            </button>
          );
        })}
      </div>
      <p className="mb-7 text-xs text-neutral-600">
        Language changes the reply, the voice and speech recognition together.
        The persona stays. Founder is still blunt in Hindi.
      </p>

      <section className="card mb-7 px-5">
        <Row label="Voice identity" hint="Used for spoken replies">
          <select
            value={prefs.voice_identity ?? VOICES[0].id}
            onChange={(e) => save({ voice_identity: e.target.value })}
            className="field w-52"
          >
            {VOICES.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
        </Row>
      </section>

      <H>Behaviour</H>
      <section className="card mb-7 px-5">
        <Row
          label="Always listening"
          hint="Sends a voice note the moment you stop speaking, without pressing send"
        >
          <Toggle
            checked={prefs.always_listening}
            onChange={(v) => save({ always_listening: v })}
          />
        </Row>
        <Row
          label="Emotion mode"
          hint="Makes responses warmer and more empathetic"
        >
          <Toggle
            checked={prefs.emotion_mode}
            onChange={(v) => save({ emotion_mode: v })}
          />
        </Row>
        <Row label="Spoken replies" hint="Read every answer aloud automatically">
          <Toggle
            checked={prefs.voice_enabled}
            onChange={(v) => save({ voice_enabled: v })}
          />
        </Row>
        <Row label="Playback speed" hint="For spoken replies">
          <div className="flex w-52 items-center gap-3">
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.25"
              value={prefs.playback_speed ?? 1}
              onChange={(e) => save({ playback_speed: Number(e.target.value) })}
            />
            <span className="w-11 shrink-0 text-right text-sm tabular-nums text-gold">
              {(prefs.playback_speed ?? 1).toFixed(2)}×
            </span>
          </div>
        </Row>
      </section>

      <H>Data</H>
      <section className="card px-5">
        <Row label="Export memories" hint="Downloads everything Eka has saved, as JSON">
          <button onClick={exportMemories} disabled={busy} className="btn-ghost">
            Download
          </button>
        </Row>
        <Row
          label="Clear conversation history"
          hint="Starts fresh chats in all four modes. Memories are kept."
        >
          <button onClick={clearHistory} className="btn-ghost">
            Clear
          </button>
        </Row>
        <Row
          label="Delete all memories"
          hint="Permanent. Eka forgets everything you have told it."
        >
          <button
            onClick={clearMemories}
            disabled={busy}
            className="rounded-lg border border-red-500/40 px-3 py-2 text-sm text-red-300 transition-all duration-200 hover:bg-red-500/10"
          >
            Delete
          </button>
        </Row>
      </section>

      <p className="mt-6 text-xs text-neutral-600">
        There is no login. Your identity is a browser-local id, so clearing site
        data starts a new user with no memories. Export first if you care about
        them.
      </p>
    </Shell>
  );
}

const H = ({ children }) => (
  <h2 className="mb-3 text-[11px] uppercase tracking-wider text-neutral-600">
    {children}
  </h2>
);

const Shell = ({ children }) => (
  <div className="scroll-thin h-full overflow-y-auto px-8 py-7">
    <div className="mx-auto max-w-2xl">{children}</div>
  </div>
);
