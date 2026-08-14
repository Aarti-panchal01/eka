import { useState } from "react";

import ekaAPI from "@/api/ekaClient";
import {
  Empty,
  ErrorBox,
  Loading,
  PageHeader,
  errText,
  useAsync,
  userId,
} from "@/lib/ui";

const MOODS = ["rough", "flat", "steady", "good", "sharp"];

const day = (iso) => {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? String(iso).slice(0, 10)
    : d.toLocaleDateString(undefined, {
        weekday: "short",
        day: "numeric",
        month: "short",
      });
};

export default function Reflections() {
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState(null);
  const { data, loading, error, reload } = useAsync(
    () => ekaAPI.getReflections(userId(), { limit: 40 }),
    []
  );
  const items = Array.isArray(data) ? data : (data?.items ?? []);

  async function create(form) {
    try {
      await ekaAPI.createReflection({
        user_id: userId(),
        content: form.content,
        mood: form.mood,
        date: new Date().toISOString().slice(0, 10),
      });
      setOpen(false);
      reload();
    } catch (err) {
      setNote(errText(err));
    }
  }

  return (
    <div className="scroll-thin h-full overflow-y-auto px-8 py-7">
      <div className="mx-auto max-w-3xl">
        <PageHeader
          title="Reflections"
          subtitle="A daily record. Eka reads these back when it matters."
          action={
            <button onClick={() => setOpen((v) => !v)} className="btn-gold">
              {open ? "Cancel" : "New reflection"}
            </button>
          }
        />

        {open && <ReflectionForm onSubmit={create} />}
        {note && <p className="mb-4 text-sm text-red-300">{note}</p>}
        {error && <ErrorBox onRetry={reload}>{error}</ErrorBox>}
        {loading && !error && <Loading label="Loading reflections…" />}
        {!loading && !error && items.length === 0 && (
          <Empty>Nothing yet. One honest paragraph is enough.</Empty>
        )}

        <div className="space-y-3">
          {items.map((r) => (
            <article key={r.id} className="card p-4">
              <div className="mb-2 flex items-center gap-2">
                <span className="text-xs text-neutral-500">
                  {day(r.date || r.created_at)}
                </span>
                {r.mood && <span className="chip">{r.mood}</span>}
              </div>
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-neutral-300">
                {r.content}
              </p>
              {r.insight && (
                <p className="mt-3 border-l-2 border-gold/40 pl-3 text-sm italic text-neutral-400">
                  {r.insight}
                </p>
              )}
            </article>
          ))}
        </div>
      </div>
    </div>
  );
}

function ReflectionForm({ onSubmit }) {
  const [content, setContent] = useState("");
  const [mood, setMood] = useState("steady");

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (content.trim()) onSubmit({ content: content.trim(), mood });
      }}
      className="card mb-5 space-y-3 p-4"
    >
      <textarea
        autoFocus
        rows={4}
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder="What actually happened today, and what you made of it."
        className="field resize-y"
      />
      <div className="flex items-center gap-2">
        <div className="flex gap-1.5">
          {MOODS.map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMood(m)}
              className={`chip transition ${
                mood === m ? "border-gold/50 text-gold" : "hover:text-neutral-200"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
        <button type="submit" className="btn-gold ml-auto">
          Save
        </button>
      </div>
    </form>
  );
}
