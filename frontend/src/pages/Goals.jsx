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

const CATEGORIES = ["business", "health", "craft", "relationships", "money"];

export default function Goals() {
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState(null);
  const { data, loading, error, reload } = useAsync(
    () => ekaAPI.getGoals(userId(), "active"),
    []
  );
  const goals = Array.isArray(data) ? data : (data?.items ?? []);

  async function create(form) {
    try {
      await ekaAPI.createGoal({
        user_id: userId(),
        title: form.title,
        category: form.category,
        target_value: Number(form.target) || 100,
        current_value: 0,
        status: "active",
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
      Math.min((goal.current_value ?? 0) + delta, goal.target_value ?? 100)
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
          subtitle="What you said you were going to do."
          action={
            <button onClick={() => setOpen((v) => !v)} className="btn-gold">
              {open ? "Cancel" : "New goal"}
            </button>
          }
        />

        {open && <GoalForm onSubmit={create} />}
        {note && <p className="mb-4 text-sm text-red-300">{note}</p>}
        {error && <ErrorBox onRetry={reload}>{error}</ErrorBox>}
        {loading && !error && <Loading label="Loading goals…" />}
        {!loading && !error && goals.length === 0 && (
          <Empty>No active goals. Start one — Eka will hold you to it.</Empty>
        )}

        <div className="space-y-3">
          {goals.map((g) => {
            const target = g.target_value || 100;
            const current = g.current_value ?? 0;
            const pct = Math.min(100, Math.round((current / target) * 100));
            return (
              <article key={g.id} className="card p-4">
                <div className="mb-3 flex items-center gap-3">
                  <h3 className="text-sm font-medium text-neutral-100">{g.title}</h3>
                  {g.category && <span className="chip">{g.category}</span>}
                  <span className="ml-auto text-sm tabular-nums text-gold">{pct}%</span>
                </div>

                <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink">
                  <div
                    className="h-full rounded-full bg-gold transition-all duration-500"
                    style={{ width: `${pct}%` }}
                  />
                </div>

                <div className="mt-3 flex items-center gap-2 text-xs text-neutral-500">
                  <span className="tabular-nums">
                    {current} / {target}
                  </span>
                  <button
                    onClick={() => bump(g, -1)}
                    className="btn-ghost ml-auto !px-2 !py-0.5"
                  >
                    −
                  </button>
                  <button onClick={() => bump(g, 1)} className="btn-ghost !px-2 !py-0.5">
                    +
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function GoalForm({ onSubmit }) {
  const [title, setTitle] = useState("");
  const [category, setCategory] = useState(CATEGORIES[0]);
  const [target, setTarget] = useState("100");

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (title.trim()) onSubmit({ title: title.trim(), category, target });
      }}
      className="card mb-5 space-y-3 p-4"
    >
      <input
        autoFocus
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Ship the thing you keep not shipping"
        className="field"
      />
      <div className="flex gap-2">
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="field flex-1"
        >
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <input
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          type="number"
          min="1"
          className="field w-28"
          title="Target value"
        />
        <button type="submit" className="btn-gold">
          Create
        </button>
      </div>
    </form>
  );
}
