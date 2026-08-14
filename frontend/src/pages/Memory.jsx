import { useRef, useState } from "react";

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

/**
 * Priority comes back two ways: `user_priority` is what the user set, and
 * `importance` (1-10) is what the pipeline inferred. User intent wins.
 */
function priorityOf(m) {
  const explicit = (m.user_priority || "").toLowerCase();
  if (explicit === "high") return { label: "High", mark: "⭐", cls: "border-gold/50 text-gold" };
  if (explicit === "low") return { label: "Low", mark: "↓", cls: "border-edge text-neutral-600" };
  if (explicit === "normal") return { label: "Normal", mark: "•", cls: "border-edge text-neutral-400" };
  const n = m.importance ?? 5;
  if (n >= 7) return { label: "High", mark: "⭐", cls: "border-gold/50 text-gold" };
  if (n <= 3) return { label: "Low", mark: "↓", cls: "border-edge text-neutral-600" };
  return { label: "Normal", mark: "•", cls: "border-edge text-neutral-400" };
}

const sourceLabel = (s) =>
  s === "upload" || s === "file" ? "📄 uploaded file" : "💬 from conversation";

/** Wrap query matches so a search result shows WHY it matched. */
function highlight(text, query) {
  const q = query.trim();
  if (!q) return text;
  const terms = q.split(/\s+/).filter((t) => t.length > 2);
  if (!terms.length) return text;
  const re = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  return text.split(re).map((part, i) =>
    re.test(part) && terms.some((t) => part.toLowerCase() === t.toLowerCase()) ? (
      <mark key={i} className="rounded bg-gold/25 px-0.5 text-gold">
        {part}
      </mark>
    ) : (
      part
    )
  );
}

export default function Memory() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);
  const [uploading, setUploading] = useState(0);
  const [preview, setPreview] = useState(null);
  const fileRef = useRef(null);

  // getMemories returns an envelope; searchMemories returns a bare array.
  const { data, loading, error, reload } = useAsync(
    () => ekaAPI.getMemories(userId(), { limit: 50 }),
    []
  );
  const all = data?.items ?? [];
  const shown = results ?? all;

  async function search(e) {
    e.preventDefault();
    if (!query.trim()) {
      setResults(null);
      return;
    }
    setBusy(true);
    setNote(null);
    try {
      const hits = await ekaAPI.searchMemories(query.trim(), userId(), 30);
      setResults(Array.isArray(hits) ? hits : (hits?.items ?? []));
    } catch (err) {
      setNote(errText(err));
    } finally {
      setBusy(false);
    }
  }

  async function upload(file) {
    if (!file) return;
    setNote(null);
    setPreview(null);
    setUploading(15); // the API gives no progress events, so this is a paced hint
    const tick = setInterval(() => setUploading((p) => Math.min(p + 10, 85)), 400);
    try {
      const saved = await ekaAPI.uploadFile(file, userId(), {
        title: file.name,
        importance: 6,
      });
      setUploading(100);
      setPreview({
        name: file.name,
        text: (saved?.content || saved?.text || "").slice(0, 400),
      });
      setResults(null);
      reload();
    } catch (err) {
      setNote(errText(err));
    } finally {
      clearInterval(tick);
      setTimeout(() => setUploading(0), 600);
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
          title="Knowledge"
          subtitle={
            results
              ? `${shown.length} result${shown.length === 1 ? "" : "s"} for "${query}"`
              : `${all.length} saved`
          }
          action={
            <>
              <input
                ref={fileRef}
                type="file"
                accept=".pdf,.txt,.md,.markdown"
                className="hidden"
                onChange={(e) => upload(e.target.files?.[0])}
              />
              <button
                onClick={() => fileRef.current?.click()}
                disabled={uploading > 0}
                className="btn-gold"
              >
                {uploading > 0 ? "Uploading…" : "Upload file"}
              </button>
            </>
          }
        />

        <Onboard title="This is Eka's memory of you">
          Everything you tell Eka gets saved here automatically. Upload a file
          and Eka reads it for future conversations. Search by meaning, not
          keywords — "the pricing argument" finds it even if you never used
          that word.
        </Onboard>

        {uploading > 0 && (
          <div className="mb-4 h-1.5 w-full overflow-hidden rounded-full bg-card">
            <div
              className="h-full rounded-full bg-gold transition-all duration-300"
              style={{ width: `${uploading}%` }}
            />
          </div>
        )}

        {preview && (
          <div className="card mb-4 p-4">
            <p className="mb-1 text-sm font-medium text-gold">
              📄 {preview.name} — read and saved
            </p>
            <p className="line-clamp-4 text-sm text-neutral-400">
              {preview.text || "Text extracted. Eka can reference this now."}
            </p>
          </div>
        )}

        <form onSubmit={search} className="mb-5 flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by meaning — try “what did I say about pricing”"
            className="field flex-1"
          />
          <button type="submit" disabled={busy} className="btn-ghost">
            {busy ? "Searching…" : "Search"}
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

        {note && <ErrorBox>{note}</ErrorBox>}
        {error && <ErrorBox onRetry={reload}>{error}</ErrorBox>}
        {loading && !error && <Loading label="Loading memories…" />}

        {!loading && !error && shown.length === 0 && (
          <Empty>
            {results ? (
              "Nothing matched that."
            ) : (
              <>
                Nothing saved yet.
                <br />
                <span className="text-sm text-neutral-600">
                  Tell Eka something about yourself in a full sentence, or
                  upload a file — short replies like "ok" are skipped on
                  purpose.
                </span>
              </>
            )}
          </Empty>
        )}

        <div className="grid gap-3 md:grid-cols-2">
          {shown.map((m) => {
            const p = priorityOf(m);
            const body = m.content || "";
            return (
              <article key={m.id} className="card p-4">
                <div className="mb-2 flex items-center gap-2">
                  <span className={`chip ${p.cls}`}>
                    {p.mark} {p.label}
                  </span>
                  <span className="chip">{sourceLabel(m.source)}</span>
                  <button
                    onClick={() => remove(m.id)}
                    title="Delete this memory"
                    className="ml-auto text-neutral-600 transition-all duration-200 hover:text-red-400"
                  >
                    🗑
                  </button>
                </div>

                {m.title && (
                  <h3 className="mb-1 text-sm font-semibold text-neutral-100">
                    {highlight(m.title, results ? query : "")}
                  </h3>
                )}
                <p className="text-sm leading-relaxed text-neutral-400">
                  {highlight(body.slice(0, 160), results ? query : "")}
                  {body.length > 160 && "…"}
                </p>

                <div className="mt-3 flex items-center gap-2 text-[11px] text-neutral-600">
                  <span>{fmtDate(m.created_date)}</span>
                  {m.score != null && (
                    <span className="text-gold/70">
                      · {Math.round(m.score * 100)}% match
                    </span>
                  )}
                  {m.topic && <span>· {m.topic}</span>}
                </div>
              </article>
            );
          })}
        </div>
      </div>
    </div>
  );
}
