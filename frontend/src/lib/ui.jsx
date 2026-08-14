import { useCallback, useEffect, useRef, useState } from "react";

import { EkaApiError, getOrCreateUserId } from "@/api/ekaClient";

/**
 * The four personas. Each owns a colour, and every Tailwind class is written
 * out in full — Tailwind scans source text, so a runtime-built string like
 * `border-${c}-500` is never emitted into the stylesheet.
 */
export const MODES = [
  {
    id: "founder",
    label: "Founder",
    hint: "Brutally honest operator",
    icon: "⚡",
    accent: "text-gold",
    dot: "bg-gold",
    // Founder is the only mode with a filled card — it is the default.
    selected: "border-gold bg-gold text-ink shadow-[0_0_24px_-6px_#f5a623]",
    selectedText: "text-ink",
    selectedHint: "text-ink/70",
    idle: "hover:border-gold/40",
  },
  {
    id: "chanakya",
    label: "Chanakya",
    hint: "Strategy and leverage",
    icon: "🛡",
    accent: "text-orange-400",
    dot: "bg-orange-400",
    selected: "border-orange-400 bg-orange-400/10 shadow-[0_0_24px_-10px_#fb923c]",
    selectedText: "text-orange-300",
    selectedHint: "text-orange-200/50",
    idle: "hover:border-orange-400/40",
  },
  {
    id: "gita",
    label: "Gita",
    hint: "Meaning when it hurts",
    icon: "✨",
    accent: "text-purple-400",
    dot: "bg-purple-400",
    selected: "border-purple-400 bg-purple-400/10 shadow-[0_0_24px_-10px_#c084fc]",
    selectedText: "text-purple-300",
    selectedHint: "text-purple-200/50",
    idle: "hover:border-purple-400/40",
  },
  {
    id: "reflection",
    label: "Reflection",
    hint: "Turns questions back",
    icon: "👁",
    accent: "text-teal-400",
    dot: "bg-teal-400",
    selected: "border-teal-400 bg-teal-400/10 shadow-[0_0_24px_-10px_#2dd4bf]",
    selectedText: "text-teal-300",
    selectedHint: "text-teal-200/50",
    idle: "hover:border-teal-400/40",
  },
];

export const modeOf = (id) => MODES.find((m) => m.id === id) ?? MODES[0];

export const NAV = [
  { to: "/", label: "Chat", icon: "💬", hint: "Talk to Eka", end: true },
  { to: "/memory", label: "Knowledge", icon: "🧠", hint: "Your memory bank" },
  { to: "/goals", label: "Goals", icon: "🎯", hint: "Track what matters" },
  { to: "/reflections", label: "Reflections", icon: "✨", hint: "Daily check-in" },
  { to: "/settings", label: "Settings", icon: "⚙️", hint: "Customize Eka" },
];

/**
 * The three languages supported end to end. Each one costs work in three
 * places — persona prompt, TTS voice, STT — so this list is deliberately short
 * rather than "every language Sarvam accepts".
 */
export const LANGUAGES = [
  { id: "en-IN", short: "EN", flag: "🇬🇧", label: "English" },
  { id: "hi-IN", short: "HI", flag: "🇮🇳", label: "Hindi" },
  { id: "kn-IN", short: "KN", flag: "🇮🇳", label: "Kannada" },
];
export const langOf = (id) => LANGUAGES.find((l) => l.id === id) ?? LANGUAGES[0];

const LANG_KEY = "eka_language";
export const getLanguage = () => {
  try {
    const v = localStorage.getItem(LANG_KEY);
    return LANGUAGES.some((l) => l.id === v) ? v : "en-IN";
  } catch {
    return "en-IN";
  }
};
export const setLanguage = (id) => {
  try {
    localStorage.setItem(LANG_KEY, id);
  } catch {
    /* private browsing */
  }
};

/**
 * One session per mode, persisted.
 *
 * Sessions live in localStorage rather than component state so switching modes
 * — or reloading — returns to that mode's conversation instead of silently
 * starting a new one. The backend takes `mode` per message, not per session,
 * so a session can legitimately outlive a mode switch.
 */
const sessionKey = (mode) => `eka_session_${mode}`;
export const getSession = (mode) => {
  try {
    return localStorage.getItem(sessionKey(mode)) || null;
  } catch {
    return null;
  }
};
export const setSession = (mode, id) => {
  try {
    if (id) localStorage.setItem(sessionKey(mode), id);
    else localStorage.removeItem(sessionKey(mode));
  } catch {
    /* private browsing */
  }
};

export const errText = (err) =>
  err instanceof EkaApiError
    ? `${err.message}${err.status ? ` (${err.status})` : ""}`
    : err?.message || "Something went wrong";

export const userId = () => getOrCreateUserId();

/**
 * Load-once async state. `reload` is stable so it is safe in a dep array, and
 * results are dropped after unmount — on Render's free tier a first request can
 * take 50s, long enough for a user to navigate away twice.
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

/** Plain-English explainer at the top of a page. */
export function Onboard({ title, children }) {
  return (
    <div className="mb-6 rounded-xl border border-gold/25 bg-gold-soft p-4">
      <p className="mb-1 text-sm font-semibold text-gold">{title}</p>
      <p className="text-sm leading-relaxed text-neutral-300">{children}</p>
    </div>
  );
}

/** Dates arrive as ISO strings; show something a person can read. */
export function fmtDate(value) {
  if (!value) return "";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value).slice(0, 10);
  return d.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: d.getFullYear() === new Date().getFullYear() ? undefined : "numeric",
  });
}

/** "Thursday, August 14" — for entries where the day itself is the headline. */
export function fmtLongDate(value) {
  if (!value) return "";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value).slice(0, 10);
  return d.toLocaleDateString(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
  });
}

export function daysUntil(value) {
  if (!value) return null;
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return null;
  return Math.ceil((d - new Date()) / 86400000);
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

export const Empty = ({ children }) => (
  <div className="card p-10 text-center text-neutral-500">{children}</div>
);

export const Loading = ({ label = "Loading…" }) => (
  <div className="card p-10 text-center text-neutral-500">
    {label}
    <p className="mt-2 text-xs text-neutral-600">
      First request after idle can take ~50s while the backend wakes up.
    </p>
  </div>
);

export const ErrorBox = ({ children, onRetry }) => (
  <div className="mb-4 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
    {children}
    {onRetry && (
      <button onClick={onRetry} className="btn-ghost ml-3 !py-1">
        Retry
      </button>
    )}
  </div>
);

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
      className={`h-6 w-11 rounded-full border transition-all duration-200 ${
        checked ? "border-gold bg-gold" : "border-edge bg-card"
      }`}
    >
      <span
        className={`block h-4 w-4 rounded-full bg-ink transition-transform duration-200 ${
          checked ? "translate-x-6" : "translate-x-1"
        }`}
      />
    </button>
  );
}
