import { useState } from "react";

import ekaAPI from "@/api/ekaClient";
import {
  Empty,
  ErrorBox,
  Loading,
  Onboard,
  PageHeader,
  daysUntil,
  errText,
  useAsync,
  userId,
} from "@/lib/ui";

const CATEGORIES = [
  { id: "startup", label: "Startup", icon: "🚀" },
  { id: "academic", label: "Academic", icon: "📚" },
  { id: "health", label: "Health", icon: "🌱" },
  { id: "personal", label: "Personal", icon: "🧭" },
  { id: "finance", label: "Finance", icon: "💰" },
];
const catOf = (id) =>
  CATEGORIES.find((c) => c.id === (id || "").toLowerCase()) ?? {
    label: id || "general",
    icon: "🎯",
  };

/**
 * A line of founder-voice commentary, derived locally from the numbers.
 *
 * Deliberately not an LLM call: this renders for every goal on every load, and
 * a per-card round trip would make the page slow and expensive for a sentence
 * that is fully determined by progress and time left.
 */
function commentary(pct, days) {
  if (pct >= 100) return "Done. Set the next one before the momentum goes.";
  if (days != null && days < 0) return "Past due. Move the date or drop it, don't leave it rotting.";
  if (days != null && days <= 3 && pct < 60) return `${days}d left and ${pct}% done. That's a scramble.`;
  if (pct === 0) return "Nothing logged yet. First increment is the hard one.";
  if (pct < 30) return "Early. Keep the cadence, not the intensity.";
  if (pct < 70) return "Halfway is where these usually stall. Don't.";
  return "Close. Finish it before starting anything new.";
}

export default function Goals() {
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState(null);
  const { data, loading, error, reload } = useAsync(
    () => ekaAPI.getGoals(userId(), "active"),
    []
  );
  const goals = Array.isArray(data) ? data : (data?.items ?? []);

  async function create(form) {
    setNote(null);
    try {
      await ekaAPI.createGoal({
        user_id: userId(),
        goal_name: form.name,
        category: form.category,
        target_value: Number(form.target) || 1,
        current_value: 0,
        unit: form.unit || "completion",
        target_date: form.date ? new Date(form.date).toISOString() : null,
      });
      setOpen(false);
      reload();
    } catch (err) {
      setNote(errText(err));
    }
  }

  async function bump(goal, delta) {
    const next = Math.max(
      0,
      Math.min((goal.current_value ?? 0) + delta, goal.target_value ?? 1)
    );
    try {
      await ekaAPI.updateGoalProgress(goal.id, next);
      reload();
    } catch (err) {
      setNote(errText(err));
    }
  }

  return (
    <div className="scroll-thin h-full overflow-y-auto px-8 py-7">
      <div className="mx-auto max-w-4xl">
        <PageHeader
          title="Goals"
          subtitle={`${goals.length} active`}
          action={
            <button onClick={() => setOpen((v) => !v)} className="btn-gold">
              {open ? "Cancel" : "New goal"}
            </button>
          }
        />

        <Onboard title="Eka holds you to these">
          Tell Eka what you're working towards. It tracks progress and brings
          goals up in conversation when they're relevant, so you get asked
          about the thing you've been avoiding.
        </Onboard>

        {open && <GoalForm onSubmit={create} />}
        {note && <ErrorBox>{note}</ErrorBox>}
        {error && <ErrorBox onRetry={reload}>{error}</ErrorBox>}
        {loading && !error && <Loading label="Loading goals…" />}
        {!loading && !error && goals.length === 0 && (
          <Empty>No active goals. Start one and Eka will keep raising it.</Empty>
        )}

        <div className="space-y-3">
          {goals.map((g) => {
            const target = g.target_value || 1;
            const current = g.current_value ?? 0;
            const pct = Math.min(100, Math.round((current / target) * 100));
            const cat = catOf(g.category);
            const days = daysUntil(g.target_date);
            const step = Math.max(1, Math.round(target / 10));
            return (
              <article key={g.id} className="card p-5">
                <div className="mb-3 flex items-center gap-3">
                  <span className="text-lg">{cat.icon}</span>
                  <h3 className="text-sm font-semibold text-neutral-100">
                    {g.goal_name}
                  </h3>
                  <span className="chip">{cat.label}</span>
                  {g.streak_days > 0 && (
                    <span className="chip border-gold/40 text-gold">
                      🔥 {g.streak_days}d streak
                    </span>
                  )}
                  <span className="ml-auto text-sm font-semibold tabular-nums text-gold">
                    {pct}%
                  </span>
                </div>

                <div className="h-2 w-full overflow-hidden rounded-full bg-ink">
                  <div
                    className="h-full rounded-full bg-gold transition-all duration-500"
                    style={{ width: `${pct}%` }}
                  />
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-neutral-500">
                  <span className="tabular-nums">
                    {current} / {target} {g.unit}
                  </span>
                  {days != null && (
                    <span className={days < 0 ? "text-red-400" : ""}>
                      {days < 0
                        ? `${Math.abs(days)}d overdue`
                        : days === 0
                          ? "due today"
                          : `${days}d left`}
                    </span>
                  )}
                  <div className="ml-auto flex items-center gap-1.5">
                    <button onClick={() => bump(g, -step)} className="btn-ghost !px-2.5 !py-1">
                      −
                    </button>
                    <button onClick={() => bump(g, step)} className="btn-ghost !px-2.5 !py-1">
                      +
                    </button>
                    <button onClick={() => bump(g, step)} className="btn-gold !px-3 !py-1">
                      Quick update
                    </button>
                  </div>
                </div>

                <p className="mt-3 border-l-2 border-gold/40 pl-3 text-xs italic text-neutral-400">
                  {commentary(pct, days)}
                </p>
              </article>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function GoalForm({ onSubmit }) {
  const [name, setName] = useState("");
  const [category, setCategory] = useState(CATEGORIES[0].id);
  const [target, setTarget] = useState("10");
  const [unit, setUnit] = useState("customers");
  const [date, setDate] = useState("");

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (name.trim()) onSubmit({ name: name.trim(), category, target, unit, date });
      }}
      className="card mb-5 space-y-3 p-4"
    >
      <input
        autoFocus
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Talk to 10 paying customers"
        className="field"
      />
      <div className="grid gap-2 sm:grid-cols-4">
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="field"
        >
          {CATEGORIES.map((c) => (
            <option key={c.id} value={c.id}>
              {c.icon}  {c.label}
            </option>
          ))}
        </select>
        <input
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          type="number"
          min="1"
          className="field"
          title="Target number"
        />
        <input
          value={unit}
          onChange={(e) => setUnit(e.target.value)}
          placeholder="customers"
          className="field"
          title="Unit"
        />
        <input
          value={date}
          onChange={(e) => setDate(e.target.value)}
          type="date"
          className="field"
          title="Due date"
        />
      </div>
      <button type="submit" className="btn-gold w-full sm:w-auto">
        Create goal
      </button>
    </form>
  );
}
