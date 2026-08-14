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

// Labels are ours; the values are what the backend's voice config expects.
const VOICES = [
  { id: "male_deep", label: "Male · Deep" },
  { id: "male_warm", label: "Male · Warm" },
  { id: "female_serene", label: "Female · Serene" },
  { id: "alien_ethereal", label: "Alien · Ethereal" },
];

const SPEEDS = [0.75, 1.0, 1.25, 1.5];

export default function Settings({ mode, onMode }) {
  const { data, loading, error, reload } = useAsync(
    () => ekaAPI.getPreferences(userId()),
    []
  );
  const [prefs, setPrefs] = useState({});
  const [saving, setSaving] = useState(false);
  const [note, setNote] = useState(null);

  useEffect(() => {
    if (data) setPrefs(data);
  }, [data]);

  // The backend is the source of truth for which voices are actually
  // configured; a Sarvam key that is missing means none of them work.
  const { data: voiceInfo } = useAsync(() => ekaAPI.getVoices().catch(() => null), []);

  async function save(patch) {
    const next = { ...prefs, ...patch };
    setPrefs(next);
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

  if (loading) return <Shell><Loading label="Loading preferences…" /></Shell>;
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
      {note && <p className="mb-4 text-sm text-red-300">{note}</p>}

      <section className="card px-5">
        <Row label="Wisdom mode" hint="Which persona new conversations open in">
          <select
            value={mode}
            onChange={(e) => onMode(e.target.value)}
            className="field w-52"
          >
            {MODES.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </Row>

        <Row
          label="Voice identity"
          hint={
            voiceInfo && voiceInfo.configured === false
              ? "Voice is not configured on the backend — selection has no effect yet"
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

        <Row label="Spoken replies" hint="Read Eka's answers aloud">
          <Toggle
            checked={prefs.voice_enabled}
            onChange={(v) => save({ voice_enabled: v })}
          />
        </Row>

        <Row
          label="Always listening"
          hint="Send automatically after a voice note, without pressing send"
        >
          <Toggle
            checked={prefs.always_listening}
            onChange={(v) => save({ always_listening: v })}
          />
        </Row>

        <Row label="Emotion mode" hint="Read sentiment and adapt tone">
          <Toggle
            checked={prefs.emotion_mode}
            onChange={(v) => save({ emotion_mode: v })}
          />
        </Row>

        <Row label="Playback speed" hint="For spoken replies">
          <div className="flex gap-1.5">
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => save({ playback_speed: s })}
                className={`chip transition ${
                  (prefs.playback_speed ?? 1) === s
                    ? "border-gold/50 text-gold"
                    : "hover:text-neutral-200"
                }`}
              >
                {s}×
              </button>
            ))}
          </div>
        </Row>
      </section>

      <p className="mt-6 text-xs text-neutral-600">
        Identity is a browser-local id — there is no login. Clearing site data
        starts a new user with no memories.
      </p>
    </Shell>
  );
}

const Shell = ({ children }) => (
  <div className="scroll-thin h-full overflow-y-auto px-8 py-7">
    <div className="mx-auto max-w-2xl">{children}</div>
  </div>
);
