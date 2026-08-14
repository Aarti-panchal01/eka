import { NavLink } from "react-router-dom";

import { MODES, NAV } from "@/lib/ui";

/** Persistent 240px left rail: identity, persona cards, section nav. */
export default function Sidebar({ mode, onMode, health, onNewChat }) {
  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-edge bg-sidebar">
      <div className="flex items-center gap-2.5 px-5 pb-4 pt-5">
        <span className="text-lg text-gold">◆</span>
        <span className="text-lg font-semibold tracking-tight">Eka</span>
        <span
          title={health === "up" ? "Connected" : "Backend unreachable"}
          className={`ml-auto h-2 w-2 rounded-full ${
            health === "up"
              ? "animate-pulse bg-emerald-400"
              : health === "down"
                ? "bg-red-400"
                : "bg-neutral-600"
          }`}
        />
      </div>

      {/* Clears only the CURRENT mode's session — the other three keep theirs. */}
      <div className="px-3 pb-4">
        <button
          onClick={onNewChat}
          className="w-full rounded-xl border border-edge bg-card px-3.5 py-2.5 text-sm text-neutral-300 transition-all duration-200 hover:border-gold/40 hover:text-gold"
        >
          ＋ New chat
        </button>
      </div>

      <p className="px-5 pb-2 text-[11px] uppercase tracking-wider text-neutral-600">
        Wisdom mode
      </p>
      <div className="space-y-2 px-3">
        {MODES.map((m) => {
          const active = m.id === mode;
          return (
            <button
              key={m.id}
              onClick={() => onMode(m.id)}
              className={`w-full rounded-xl border px-3.5 py-3 text-left transition-all duration-200 ${
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
                className={`mt-1 pl-7 text-[11px] leading-snug ${
                  active ? m.selectedHint : "text-neutral-500"
                }`}
              >
                {m.hint}
              </p>
            </button>
          );
        })}
      </div>

      <nav className="mt-7 space-y-1 px-3 pb-5">
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.end}
            className={({ isActive }) =>
              `flex items-start gap-3 rounded-lg px-3.5 py-2.5 transition-all duration-200 ${
                isActive
                  ? "bg-card text-neutral-100"
                  : "text-neutral-500 hover:bg-card/60 hover:text-neutral-200"
              }`
            }
          >
            <span className="mt-0.5 text-base leading-none">{n.icon}</span>
            <span className="min-w-0">
              <span className="block text-sm leading-tight">{n.label}</span>
              <span className="block text-[11px] leading-tight text-neutral-600">
                {n.hint}
              </span>
            </span>
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
