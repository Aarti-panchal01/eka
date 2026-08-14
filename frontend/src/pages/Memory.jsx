import { useRef, useState } from "react";

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

const PRIORITY = [
  { min: 7, label: "High", cls: "border-gold/50 text-gold" },
  { min: 4, label: "Normal", cls: "border-edge text-neutral-400" },
  { min: 0, label: "Low", cls: "border-edge text-neutral-600" },
];
const tagFor = (n) => PRIORITY.find((p) => (n ?? 5) >= p.min) ?? PRIORITY[1];

export default function Memory() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null); // null = browsing, [] = no hits
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);
  const fileRef = useRef(null);

  // getMemories returns an envelope; searchMemories returns a bare array.
  const { data, loading, error, reload } = useAsync(
    () => ekaAPI.getMemories(userId(), { limit: 60 }),
    []
  );
  const memories = results ?? data?.items ?? [];

  async function search(e) {
    e.preventDefault();
    if (!query.trim()) {
      setResults(null);
      return;
    }
    setBusy(true);
    setNote(null);
    try {
      setResults(await ekaAPI.searchMemories(query.trim(), userId(), 30));
    } catch (err) {
      setNote(errText(err));
    } finally {
      setBusy(false);
    }
  }

  async function upload(file) {
    if (!file) return;
    setBusy(true);
    setNote(null);
    try {
      await ekaAPI.uploadFile(file, userId(), { title: file.name, importance: 6 });
      setNote(`Uploaded ${file.name}`);
      setResults(null);
      reload();
    } catch (err) {
      setNote(errText(err));
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function remove(id) {
    try {
      await ekaAPI.deleteMemory(id, userId());
      setResults((r) => (r ? r.filter((m) => m.id !== id) : r));
      reload();
    } catch (err) {
      setNote(errText(err));
    }
  }

  return (
    <div className="scroll-thin h-full overflow-y-auto px-8 py-7">
      <div className="mx-auto max-w-4xl">
        <PageHeader
          title="Knowledge base"
          subtitle="Everything Eka remembers about you. Search is semantic, not keyword."
          action={
            <>
              <input
                ref={fileRef}
                type="file"
                className="hidden"
                onChange={(e) => upload(e.target.files?.[0])}
              />
              <button
                onClick={() => fileRef.current?.click()}
                disabled={busy}
                className="btn-gold"
              >
                Upload file
              </button>
            </>
          }
        />

        <form onSubmit={search} className="mb-5 flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search meaning, not words — “the pricing argument”"
            className="field flex-1"
          />
          <button type="submit" disabled={busy} className="btn-ghost">
            Search
          </button>
          {results && (
            <button
              type="button"
              onClick={() => {
                setResults(null);
                setQuery("");
              }}
              className="btn-ghost"
            >
              Clear
            </button>
          )}
        </form>

        {note && <p className="mb-4 text-sm text-neutral-400">{note}</p>}
        {error && <ErrorBox onRetry={reload}>{error}</ErrorBox>}
        {loading && !error && <Loading label="Loading memories…" />}

        {!loading && !error && memories.length === 0 && (
          <Empty>
            {results
              ? "Nothing matched that."
              : "No memories yet. Talk to Eka, or upload a file."}
          </Empty>
        )}

        <div className="grid gap-3 md:grid-cols-2">
          {memories.map((m) => {
            const tag = tagFor(m.priority);
            return (
              <article key={m.id} className="card p-4">
                <div className="mb-2 flex items-center gap-2">
                  <span className={`chip ${tag.cls}`}>{tag.label}</span>
                  {m.mode && <span className="chip">{m.mode}</span>}
                  {m.source_type && <span className="chip">{m.source_type}</span>}
                  <button
                    onClick={() => remove(m.id)}
                    title="Delete"
                    className="ml-auto text-xs text-neutral-600 hover:text-red-400"
                  >
                    ✕
                  </button>
                </div>
                {m.title && (
                  <h3 className="mb-1 text-sm font-medium text-neutral-200">
                    {m.title}
                  </h3>
                )}
                <p className="line-clamp-5 whitespace-pre-wrap text-sm leading-relaxed text-neutral-400">
                  {m.content || m.text}
                </p>
                {m.tags?.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {m.tags.slice(0, 5).map((t) => (
                      <span key={t} className="chip">
                        {t}
                      </span>
                    ))}
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
