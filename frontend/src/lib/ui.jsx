import { useCallback, useEffect, useRef, useState } from "react";

import { EkaApiError, getOrCreateUserId } from "@/api/ekaClient";

/** The four personas, in one place — sidebar, chat header and settings all read this. */
export const MODES = [
  { id: "founder", label: "Founder", hint: "Brutally honest operator", glyph: "◆" },
  { id: "chanakya", label: "Chanakya", hint: "Strategy and leverage", glyph: "◈" },
  { id: "gita", label: "Gita", hint: "Meaning when it hurts", glyph: "◇" },
  { id: "reflection", label: "Reflection", hint: "Turns questions back", glyph: "○" },
];

export const errText = (err) =>
  err instanceof EkaApiError
    ? `${err.message}${err.status ? ` (${err.status})` : ""}`
    : err?.message || "Something went wrong";

export const userId = () => getOrCreateUserId();

/**
 * Load-once async state with the three states a screen actually needs.
 *
 * `reload` is stable, so it is safe in a dependency array. Results are dropped
 * if the component unmounted mid-flight — on Render's free tier a first request
 * can take 50s, which is long enough for a user to navigate away twice.
 */
export function useAsync(fn, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const alive = useRef(true);

  const reload = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.resolve()
      .then(fn)
      .then((d) => alive.current && setData(d))
      .catch((e) => alive.current && setError(errText(e)))
      .finally(() => alive.current && setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    alive.current = true;
    reload();
    return () => {
      alive.current = false;
    };
  }, [reload]);

  return { data, loading, error, reload, setData };
}

export function PageHeader({ title, subtitle, action }) {
  return (
    <div className="mb-6 flex items-start justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-neutral-500">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

export function Empty({ children }) {
  return (
    <div className="card p-10 text-center text-neutral-500">{children}</div>
  );
}

export function Loading({ label = "Loading…" }) {
  return (
    <div className="card p-10 text-center text-neutral-500">
      {label}
      <p className="mt-2 text-xs text-neutral-600">
        First request after idle can take ~50s while Render wakes the backend.
      </p>
    </div>
  );
}

export function ErrorBox({ children, onRetry }) {
  return (
    <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
      {children}
      {onRetry && (
        <button onClick={onRetry} className="btn-ghost ml-3 !py-1">
          Retry
        </button>
      )}
    </div>
  );
}

/** A labelled row for Settings — label left, control right. */
export function Row({ label, hint, children }) {
  return (
    <div className="flex items-center justify-between gap-6 border-b border-edge py-4 last:border-0">
      <div className="min-w-0">
        <p className="text-sm text-neutral-200">{label}</p>
        {hint && <p className="mt-0.5 text-xs text-neutral-500">{hint}</p>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

export function Toggle({ checked, onChange }) {
  return (
    <button
      role="switch"
      aria-checked={!!checked}
      onClick={() => onChange(!checked)}
      className={`h-6 w-11 rounded-full border transition ${
        checked ? "border-gold bg-gold" : "border-edge bg-card"
      }`}
    >
      <span
        className={`block h-4 w-4 rounded-full bg-ink transition ${
          checked ? "translate-x-6" : "translate-x-1"
        }`}
      />
    </button>
  );
}
