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
  fmtLongDate,
  useAsync,
  userId,
} from "@/lib/ui";

const MOODS = [
  { id: "good", emoji: "😊", label: "Good", ring: "border-emerald-400/60", tint: "bg-emerald-400/15" },
  { id: "neutral", emoji: "😐", label: "Neutral", ring: "border-neutral-500/60", tint: "bg-neutral-500/15" },
  { id: "tough", emoji: "😔", label: "Tough", ring: "border-sky-400/60", tint: "bg-sky-400/15" },
  { id: "energized", emoji: "🔥", label: "Energized", ring: "border-gold/60", tint: "bg-gold/15" },
];

/**
 * Older entries were saved with a different vocabulary ("flat", "steady",
 * "rough", "sharp"). Mapping them here means old rows render as a real mood
 * instead of a bare word with no emoji — which is what made the page look like
 * raw database output.
 */
const MOOD_ALIASES = {
  good: "good",
  sharp: "good",
  great: "good",
  neutral: "neutral",
  flat: "neutral",
  steady: "neutral",
  ok: "neutral",
  tough: "tough",
  rough: "tough",
  bad: "tough",
  low: "tough",
  energized: "energized",
  fired: "energized",
};

const moodOf = (raw) => {
  const key = MOOD_ALIASES[(raw || "").toLowerCase().trim()];
  return (
    MOODS.find((m) => m.id === key) ?? {
      emoji: "•",
      label: raw || "Unrecorded",
      ring: "border-edge",
      tint: "bg-card",
    }
  );
};

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
        request_commentary: true, // this is what makes Eka write the insight
      });
      setOpen(false);
      reload();
    } catch (err) {
      setNote(errText(err));
    } finally {
      setSaving(false);
    }
  }

  async function remove(r) {
    // Mood-only rows are noise and go without ceremony; anything with writing
    // in it is worth one confirmation before it is gone for good.
    const hasText = [r.challenges_faced, r.learnings, r.gratitude].some(
      (v) => v && String(v).trim()
    );
    if (hasText && !window.confirm("Delete this reflection? This cannot be undone."))
      return;
    try {
      await ekaAPI.deleteReflection(r.id);
      reload();
    } catch (err) {
      setNote(errText(err));
    }
  }

  const week = [...items].slice(0, 7).reverse();

  return (
    <div className="scroll-thin h-full overflow-y-auto px-8 py-7">
      <div className="mx-auto max-w-3xl">
        <PageHeader
          title="Reflections"
          subtitle={`${items.length} ${items.length === 1 ? "entry" : "entries"}`}
          action={
            <button onClick={() => setOpen(true)} className="btn-gold">
              New reflection
            </button>
          }
        />

        <Onboard title="End the day here">
          Eka reads your reflections and brings them up when they're relevant, so
          a pattern you can't see from inside becomes visible from outside.
        </Onboard>

        {week.length > 1 && (
          <div className="card mb-5 p-5">
            <p className="mb-4 text-[11px] uppercase tracking-wider text-neutral-600">
              Recent mood
            </p>
            <div className="flex flex-wrap gap-5">
              {week.map((r) => {
                const m = moodOf(r.mood);
                return (
                  <div key={r.id} className="text-center">
                    <div
                      className={`flex h-11 w-11 items-center justify-center rounded-full border text-xl ${m.ring} ${m.tint}`}
                      title={m.label}
                    >
                      {m.emoji}
                    </div>
                    <div className="mt-1.5 text-[11px] text-neutral-400">
                      {m.label}
                    </div>
                    <div className="text-[10px] text-neutral-600">
                      {fmtDate(r.date || r.created_date)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {note && <ErrorBox>{note}</ErrorBox>}
        {error && <ErrorBox onRetry={reload}>{error}</ErrorBox>}
        {loading && !error && <Loading label="Loading reflections…" />}
        {!loading && !error && items.length === 0 && (
          <Empty>Nothing yet. One honest paragraph is enough.</Empty>
        )}

        <div className="space-y-3">
          {items.map((r) => (
            <Card key={r.id} r={r} onDelete={() => remove(r)} />
          ))}
        </div>
      </div>

      {open && (
        <Panel onClose={() => !saving && setOpen(false)}>
          <ReflectionForm onSubmit={create} saving={saving} />
        </Panel>
      )}
    </div>
  );
}

function Card({ r, onDelete }) {
  const [expanded, setExpanded] = useState(false);
  const m = moodOf(r.mood);
  const blocks = [
    ["What happened", r.challenges_faced],
    ["What I learned", r.learnings],
    ["Grateful for", r.gratitude],
  ].filter(([, v]) => v && String(v).trim());

  // Only offer expand when there is genuinely more to see.
  const long = blocks.some(([, v]) => String(v).length > 220);

  return (
    <article className="card p-5">
      <div className="mb-4 flex items-center gap-3">
        <div
          className={`flex h-10 w-10 items-center justify-center rounded-full border text-lg ${m.ring} ${m.tint}`}
        >
          {m.emoji}
        </div>
        <div>
          <p className="text-sm font-medium text-neutral-100">
            {fmtLongDate(r.date || r.created_date)}
          </p>
          <p className="text-xs text-neutral-500">{m.label}</p>
        </div>
        <button
          onClick={onDelete}
          title="Delete this reflection"
          className="ml-auto text-neutral-600 transition-all duration-200 hover:text-red-400"
        >
          🗑
        </button>
      </div>

      {blocks.length === 0 && (
        <p className="text-sm italic text-neutral-600">
          No written entry, mood only.
        </p>
      )}

      {blocks.map(([label, value]) => (
        <div key={label} className="mb-3 last:mb-0">
          <p className="mb-1 text-[11px] uppercase tracking-wider text-neutral-600">
            {label}
          </p>
          <p
            className={`whitespace-pre-wrap text-sm leading-relaxed text-neutral-300 ${
              expanded ? "" : "line-clamp-4"
            }`}
          >
            {value}
          </p>
        </div>
      ))}

      {long && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="mt-1 text-[11px] text-gold transition-all duration-200 hover:brightness-125"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}

      {r.eka_commentary && (
        <div className="mt-4 rounded-lg border border-purple-400/30 bg-purple-400/[0.07] p-4">
          <p className="mb-1 text-[11px] uppercase tracking-wider text-purple-400">
            ✨ Eka's reflection
          </p>
          <p className="text-sm leading-relaxed text-purple-100/85">
            {r.eka_commentary}
          </p>
        </div>
      )}
    </article>
  );
}

/** Slide-in panel from the right, with a scrim that closes it. */
function Panel({ children, onClose }) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div
        onClick={onClose}
        className="absolute inset-0 bg-black/60 backdrop-blur-[2px]"
      />
      <div className="animate-slideIn relative h-full w-full max-w-md overflow-y-auto border-l border-edge bg-sidebar p-6 shadow-2xl">
        <button
          onClick={onClose}
          className="absolute right-4 top-4 text-neutral-500 transition-all duration-200 hover:text-neutral-100"
        >
          ✕
        </button>
        {children}
      </div>
    </div>
  );
}

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
      className="space-y-5"
    >
      <div>
        <h2 className="text-lg font-semibold">Today's reflection</h2>
        <p className="mt-1 text-xs text-neutral-500">
          {fmtLongDate(new Date().toISOString())}
        </p>
      </div>

      <div>
        <p className="mb-2 text-[11px] uppercase tracking-wider text-neutral-600">
          How was it?
        </p>
        <div className="grid grid-cols-4 gap-2">
          {MOODS.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setMood(m.id)}
              className={`rounded-xl border py-3 transition-all duration-200 ${
                mood === m.id
                  ? `${m.ring} ${m.tint}`
                  : "border-edge bg-card hover:border-neutral-600"
              }`}
            >
              <div className="text-2xl">{m.emoji}</div>
              <div className="mt-1 text-[10px] text-neutral-400">{m.label}</div>
            </button>
          ))}
        </div>
      </div>

      <Field
        label="What happened today?"
        placeholder="The thing that actually took up your head."
        value={challenges}
        onChange={setChallenges}
        autoFocus
      />
      <Field
        label="What did I learn?"
        placeholder="Even if it's small, or annoying."
        value={learnings}
        onChange={setLearnings}
      />
      <Field
        label="What am I grateful for?"
        placeholder="One specific thing beats three vague ones."
        value={gratitude}
        onChange={setGratitude}
      />

      <button type="submit" disabled={empty || saving} className="btn-gold w-full">
        {saving ? "Saving… Eka is reading it" : "Save reflection"}
      </button>
      <p className="text-center text-[11px] text-neutral-600">
        Eka writes a one-line insight back.
      </p>
    </form>
  );
}

const Field = ({ label, placeholder, value, onChange, autoFocus }) => (
  <div>
    <label className="mb-1.5 block text-[11px] uppercase tracking-wider text-neutral-600">
      {label}
    </label>
    <textarea
      autoFocus={autoFocus}
      rows={3}
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      className="field resize-y"
    />
  </div>
);
