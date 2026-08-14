import { useState } from "react";

import ekaAPI from "@/api/ekaClient";
import {
  Empty,
  ErrorBox,
  Loading,
  Onboard,
  PageHeader,
  errText,
  fmtDate,
  useAsync,
  userId,
} from "@/lib/ui";

const MOODS = [
  { id: "good", emoji: "😊", label: "Good" },
  { id: "neutral", emoji: "😐", label: "Neutral" },
  { id: "tough", emoji: "😔", label: "Tough" },
  { id: "energized", emoji: "🔥", label: "Energized" },
];
const moodOf = (id) => MOODS.find((m) => m.id === id) ?? { emoji: "•", label: id || "" };

export default function Reflections() {
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState(null);
  const [saving, setSaving] = useState(false);
  const { data, loading, error, reload } = useAsync(
    () => ekaAPI.getReflections(userId(), { limit: 60 }),
    []
  );
  const items = Array.isArray(data) ? data : (data?.items ?? []);

  async function create(form) {
    setSaving(true);
    setNote(null);
    try {
      await ekaAPI.createReflection({
        user_id: userId(),
        date: new Date().toISOString(),
        mood: form.mood,
        challenges_faced: form.challenges,
        learnings: form.learnings,
        gratitude: form.gratitude,
        // This is what makes Eka write eka_commentary on the entry.
        request_commentary: true,
      });
      setOpen(false);
      reload();
    } catch (err) {
      setNote(errText(err));
    } finally {
      setSaving(false);
    }
  }

  // Newest first, then the last 7 reversed so the strip reads left-to-right.
  const week = [...items].slice(0, 7).reverse();

  return (
    <div className="scroll-thin h-full overflow-y-auto px-8 py-7">
      <div className="mx-auto max-w-3xl">
        <PageHeader
          title="Reflections"
          subtitle={`${items.length} entries`}
          action={
            <button onClick={() => setOpen((v) => !v)} className="btn-gold">
              {open ? "Cancel" : "New reflection"}
            </button>
          }
        />

        <Onboard title="End the day here">
          Eka reads your reflections and brings them up when they're relevant —
          so a pattern you can't see from inside becomes visible from outside.
        </Onboard>

        {week.length > 1 && (
          <div className="card mb-5 flex items-center gap-4 p-4">
            <span className="text-[11px] uppercase tracking-wider text-neutral-600">
              Recent mood
            </span>
            <div className="flex gap-4">
              {week.map((r) => (
                <div key={r.id} className="text-center" title={moodOf(r.mood).label}>
                  <div className="text-lg">{moodOf(r.mood).emoji}</div>
                  <div className="mt-0.5 text-[10px] text-neutral-600">
                    {fmtDate(r.date || r.created_date)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {open && <ReflectionForm onSubmit={create} saving={saving} />}
        {note && <ErrorBox>{note}</ErrorBox>}
        {error && <ErrorBox onRetry={reload}>{error}</ErrorBox>}
        {loading && !error && <Loading label="Loading reflections…" />}
        {!loading && !error && items.length === 0 && (
          <Empty>Nothing yet. One honest paragraph is enough.</Empty>
        )}

        <div className="space-y-3">
          {items.map((r) => {
            const mood = moodOf(r.mood);
            return (
              <article key={r.id} className="card p-5">
                <div className="mb-3 flex items-center gap-2">
                  <span className="text-lg">{mood.emoji}</span>
                  <span className="text-sm text-neutral-300">{mood.label}</span>
                  <span className="ml-auto text-xs text-neutral-600">
                    {fmtDate(r.date || r.created_date)}
                  </span>
                </div>

                {r.challenges_faced && (
                  <Block label="What happened">{r.challenges_faced}</Block>
                )}
                {r.learnings && <Block label="What I learned">{r.learnings}</Block>}
                {r.gratitude && <Block label="Grateful for">{r.gratitude}</Block>}

                {r.eka_commentary && (
                  <div className="mt-3.5 rounded-lg border border-purple-400/30 bg-purple-400/[0.07] p-3.5">
                    <p className="mb-1 text-[11px] uppercase tracking-wider text-purple-400">
                      ✨ Eka's reflection
                    </p>
                    <p className="text-sm leading-relaxed text-purple-100/80">
                      {r.eka_commentary}
                    </p>
                  </div>
                )}
              </article>
            );
          })}
        </div>
      </div>
    </div>
  );
}

const Block = ({ label, children }) => (
  <div className="mb-2.5 last:mb-0">
    <p className="mb-0.5 text-[11px] uppercase tracking-wider text-neutral-600">
      {label}
    </p>
    <p className="whitespace-pre-wrap text-sm leading-relaxed text-neutral-300">
      {children}
    </p>
  </div>
);

function ReflectionForm({ onSubmit, saving }) {
  const [mood, setMood] = useState("good");
  const [challenges, setChallenges] = useState("");
  const [learnings, setLearnings] = useState("");
  const [gratitude, setGratitude] = useState("");

  const empty = !challenges.trim() && !learnings.trim() && !gratitude.trim();

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (!empty) onSubmit({ mood, challenges, learnings, gratitude });
      }}
      className="card mb-5 space-y-4 p-5"
    >
      <div className="flex gap-2">
        {MOODS.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => setMood(m.id)}
            className={`flex-1 rounded-xl border px-3 py-2.5 transition-all duration-200 ${
              mood === m.id
                ? "border-gold bg-gold-soft text-gold"
                : "border-edge bg-ink text-neutral-400 hover:border-gold/40"
            }`}
          >
            <div className="text-lg">{m.emoji}</div>
            <div className="mt-0.5 text-[11px]">{m.label}</div>
          </button>
        ))}
      </div>

      <Field label="What happened today?" value={challenges} onChange={setChallenges} autoFocus />
      <Field label="What did I learn?" value={learnings} onChange={setLearnings} />
      <Field label="What am I grateful for?" value={gratitude} onChange={setGratitude} />

      <div className="flex items-center gap-3">
        <button type="submit" disabled={empty || saving} className="btn-gold">
          {saving ? "Saving… Eka is reading it" : "Save reflection"}
        </button>
        <span className="text-xs text-neutral-600">
          Eka writes a one-line insight back.
        </span>
      </div>
    </form>
  );
}

const Field = ({ label, value, onChange, autoFocus }) => (
  <div>
    <label className="mb-1 block text-[11px] uppercase tracking-wider text-neutral-600">
      {label}
    </label>
    <textarea
      autoFocus={autoFocus}
      rows={2}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="field resize-y"
    />
  </div>
);
