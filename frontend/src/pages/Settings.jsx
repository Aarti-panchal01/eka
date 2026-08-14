import { useEffect, useState } from "react";

import ekaAPI from "@/api/ekaClient";
import {
  ErrorBox,
  Loading,
  MODES,
  PageHeader,
  Row,
  Toggle,
  errText,
  useAsync,
  userId,
} from "@/lib/ui";

// Our labels; the values are what the backend's voice config expects.
const VOICES = [
  { id: "male_deep", label: "Male · Deep" },
  { id: "male_warm", label: "Male · Warm" },
  { id: "female_serene", label: "Female · Serene" },
  { id: "alien_ethereal", label: "Alien · Ethereal" },
];

export default function Settings({ mode, onMode }) {
  const { data, loading, error, reload } = useAsync(
    () => ekaAPI.getPreferences(userId()),
    []
  );
  const { data: voiceInfo } = useAsync(
    () => ekaAPI.getVoices().catch(() => null),
    []
  );
  const [prefs, setPrefs] = useState({});
  const [saving, setSaving] = useState(false);
  const [note, setNote] = useState(null);

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

      <h2 className="mb-3 text-[11px] uppercase tracking-wider text-neutral-600">
        Wisdom mode
      </h2>
      <div className="mb-7 grid grid-cols-2 gap-3">
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

      <section className="card px-5">
        <Row
          label="Voice identity"
          hint={
            voiceInfo && voiceInfo.configured === false
              ? "Voice is not configured on the backend yet"
              : "Used for spoken replies"
          }
        >
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

        <Row label="Always listening" hint="Send straight after a voice note">
          <Toggle
            checked={prefs.always_listening}
            onChange={(v) => save({ always_listening: v })}
          />
        </Row>

        <Row label="Emotion mode" hint="Adapt tone to how you sound">
          <Toggle
            checked={prefs.emotion_mode}
            onChange={(v) => save({ emotion_mode: v })}
          />
        </Row>

        <Row label="Spoken replies" hint="Read Eka's answers aloud">
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
              onChange={(e) =>
                save({ playback_speed: Number(e.target.value) })
              }
            />
            <span className="w-10 shrink-0 text-right text-sm tabular-nums text-gold">
              {(prefs.playback_speed ?? 1).toFixed(2)}×
            </span>
          </div>
        </Row>
      </section>

      <p className="mt-6 text-xs text-neutral-600">
        There is no login. Your identity is a browser-local id — clearing site
        data starts a new user with no memories.
      </p>
    </Shell>
  );
}

const Shell = ({ children }) => (
  <div className="scroll-thin h-full overflow-y-auto px-8 py-7">
    <div className="mx-auto max-w-2xl">{children}</div>
  </div>
);
