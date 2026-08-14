import { NavLink } from "react-router-dom";

import { MODES } from "@/lib/ui";

const NAV = [
  { to: "/", label: "Chat", glyph: "▣", end: true },
  { to: "/memory", label: "Knowledge", glyph: "▤" },
  { to: "/goals", label: "Goals", glyph: "▥" },
  { to: "/reflections", label: "Reflections", glyph: "▦" },
  { to: "/settings", label: "Settings", glyph: "▧" },
];

/**
 * Persistent left rail: identity, the four persona cards, then section nav.
 *
 * The mode cards live here rather than inside Chat because mode is app-level
 * state — Settings reads it, and the chat header reflects it.
 */
export default function Sidebar({ mode, onMode, health }) {
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-edge bg-ink">
      <div className="flex items-center gap-2.5 px-5 py-5">
        <span className="text-lg text-gold">◆</span>
        <span className="text-lg font-semibold tracking-tight">Eka</span>
        <span
          title={health === "up" ? "Backend reachable" : "Backend unreachable"}
          className={`ml-auto h-1.5 w-1.5 rounded-full ${
            health === "up"
              ? "bg-emerald-400"
              : health === "down"
                ? "bg-red-400"
                : "bg-neutral-600"
          }`}
        />
      </div>

      <p className="px-5 pb-2 text-[11px] uppercase tracking-wider text-neutral-600">
        Wisdom mode
      </p>
      <div className="scroll-thin space-y-1.5 overflow-y-auto px-3">
        {MODES.map((m) => {
          const active = m.id === mode;
          return (
            <button
              key={m.id}
              onClick={() => onMode(m.id)}
              className={`w-full rounded-lg border px-3 py-2.5 text-left transition ${
                active
                  ? "border-gold/40 bg-gold-soft"
                  : "border-transparent hover:border-edge hover:bg-card"
              }`}
            >
              <div className="flex items-center gap-2">
                <span className={active ? "text-gold" : "text-neutral-500"}>
                  {m.glyph}
                </span>
                <span
                  className={`text-sm font-medium ${
                    active ? "text-gold" : "text-neutral-200"
                  }`}
                >
                  {m.label}
                </span>
              </div>
              <p className="mt-0.5 pl-6 text-[11px] leading-snug text-neutral-500">
                {m.hint}
              </p>
            </button>
          );
        })}
      </div>

      <nav className="mt-6 space-y-0.5 px-3 pb-5">
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.end}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition ${
                isActive
                  ? "bg-card text-neutral-100"
                  : "text-neutral-500 hover:text-neutral-200"
              }`
            }
          >
            <span className="text-xs">{n.glyph}</span>
            {n.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
