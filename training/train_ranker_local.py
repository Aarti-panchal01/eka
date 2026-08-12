"""
================================================================================
EKA — memory reranker (LightGBM LambdaRank)  |  CPU, no GPU, <10 minutes
================================================================================
    python training/train_ranker_local.py

Vector search gives Eka a *pool* of candidate memories ranked purely by cosine
similarity. That's not the same as "which memories should Eka actually bring
into this reply". A memory can be semantically close but stale, trivial, or
explicitly deprioritised by the user. This model learns that trade-off.

WHY SYNTHETIC DATA
------------------
There is no click-through log on day one. So we encode the ranking policy we
actually want as a scoring function, sample features from realistic
distributions, and let LambdaRank learn a smooth version of it. Once real usage
data exists (which memories Eka used vs which the user reacted to), retrain on
that and delete the synthetic path.

THE 8 FEATURES (order matters — serving code depends on it)
-----------------------------------------------------------
    f0  cosine_similarity     0-1    from Qdrant
    f1  importance            1-10   user- or auto-assigned
    f2  priority_weight       1/2/3  low / normal / high
    f3  days_old              0-365
    f4  access_count          0-50   how often this memory has been recalled
    f5  topic_match           0/1    memory topic == inferred query topic
    f6  source_type           1/2/3  chat / upload / manual
    f7  content_length_norm   0-1    normalised length
================================================================================
"""

import json
import sys
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "ranker"
MODEL_PATH = MODEL_DIR / "eka_ranker.txt"
META_PATH = MODEL_DIR / "feature_names.json"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

FEATURE_NAMES = [
    "cosine_similarity",
    "importance",
    "priority_weight",
    "days_old",
    "access_count",
    "topic_match",
    "source_type",
    "content_length_norm",
]

N_QUERIES = 500
CANDIDATES_PER_QUERY = 10
# Roughly a third of any retrieved pool is genuinely on-topic; the rest are
# near-misses. Modelling that imbalance is the whole point.
RELEVANT_FRACTION = 0.35

rng = np.random.default_rng(42)


def make_dataset(n_queries: int) -> tuple:
    """Return (X, y, groups) for a LambdaRank problem."""
    rows, labels, groups = [], [], []

    for _ in range(n_queries):
        n_relevant = max(1, int(rng.binomial(CANDIDATES_PER_QUERY, RELEVANT_FRACTION)))
        relevance_flags = np.array(
            [True] * n_relevant + [False] * (CANDIDATES_PER_QUERY - n_relevant)
        )
        rng.shuffle(relevance_flags)

        for is_relevant in relevance_flags:
            # f0 — the dominant signal. Relevant memories cluster high.
            cosine = rng.beta(3, 2) if is_relevant else rng.beta(1, 4)
            # f1 — importance, uniform 1-10.
            importance = float(rng.integers(1, 11))
            # f2 — user priority. Most memories are "normal".
            priority = float(rng.choice([1.0, 2.0, 3.0], p=[0.15, 0.65, 0.20]))
            # f3 — age. Exponentially distributed: most memories are recent.
            days_old = float(min(365, rng.exponential(60)))
            # f4 — how often recalled before.
            access_count = float(min(50, rng.poisson(5)))
            # f5 — topic match, correlated with relevance (a relevant memory is
            # much more likely to share the query's topic).
            topic_match = float(rng.random() < (0.55 if is_relevant else 0.12))
            # f6 — provenance. Manual/uploaded memories are rarer.
            source_type = float(rng.choice([1.0, 2.0, 3.0], p=[0.7, 0.15, 0.15]))
            # f7 — normalised content length.
            length_norm = float(rng.beta(2, 5))

            recency = 1.0 - (days_old / 365.0)
            # Weighting note: the original build spec used 2.5*cosine against
            # ~0.6 for all seven other features combined. That makes the label
            # an almost-monotone function of cosine, so LambdaRank converges in
            # 3 trees, scores NDCG@3 ~0.97, and attributes >90% of gain to
            # cosine — i.e. it learns nothing the "sort by Qdrant score"
            # fallback doesn't already do for free.
            #
            # Cosine still leads (it should — it's the only feature that knows
            # what the query was about), but the rest now carry enough weight
            # to change the ordering of similarly-relevant candidates, which is
            # the entire reason this model exists.
            score = (
                1.60 * cosine
                + 0.55 * (importance / 10.0)
                + 0.50 * (priority / 3.0)
                + 0.45 * recency
                + 0.40 * topic_match
                + 0.20 * min(1.0, access_count / 20.0)
                # A very old memory is actively misleading, not merely less
                # useful — an interaction term the linear part can't express.
                - 0.35 * (1.0 if days_old > 180 else 0.0) * (1.0 - cosine)
                + rng.normal(0, 0.35)
            )
            # Rescale to 0-3 relevance grades. Max achievable score is ~3.7.
            label = int(np.clip(round(score * (3.0 / 3.2)), 0, 3))

            rows.append([
                cosine, importance, priority, days_old,
                access_count, topic_match, source_type, length_norm,
            ])
            labels.append(label)

        groups.append(CANDIDATES_PER_QUERY)

    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int32), \
        np.asarray(groups, dtype=np.int32)


def main() -> None:
    try:
        import lightgbm as lgb
    except ImportError:
        sys.exit("pip install lightgbm==4.3.0 numpy")

    print("EKA memory reranker — LightGBM LambdaRank\n")

    X, y, groups = make_dataset(N_QUERIES)
    n_train_queries = int(len(groups) * 0.8)
    train_rows = int(groups[:n_train_queries].sum())

    X_train, y_train, train_groups = X[:train_rows], y[:train_rows], groups[:n_train_queries]
    X_val, y_val, val_groups = X[train_rows:], y[train_rows:], groups[n_train_queries:]

    print(f"  train: {len(X_train)} rows / {len(train_groups)} queries")
    print(f"  val  : {len(X_val)} rows / {len(val_groups)} queries")
    print(f"  label distribution: "
          f"{dict(zip(*[a.tolist() for a in np.unique(y, return_counts=True)]))}\n")

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        group=train_groups,
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        eval_at=[3, 5],
        feature_name=FEATURE_NAMES,
        # Patience of 50, not 20: NDCG plateaus in noisy stretches early on and
        # a short patience stops before the non-cosine features get used.
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )

    ndcg3 = model.best_score_["valid_0"]["ndcg@3"]
    ndcg5 = model.best_score_["valid_0"]["ndcg@5"]
    print(f"\n  NDCG@3 {ndcg3:.4f}   NDCG@5 {ndcg5:.4f}   (target >0.70)")
    print(f"  best iteration: {model.best_iteration_}")

    print("\n  feature importance (gain):")
    importances = model.booster_.feature_importance(importance_type="gain")
    total = importances.sum() or 1.0
    for name, gain in sorted(
        zip(FEATURE_NAMES, importances), key=lambda kv: -kv[1]
    ):
        bar = "█" * int(30 * gain / total)
        print(f"    {name:<22}{100 * gain / total:>6.1f}%  {bar}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH))
    META_PATH.write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "ndcg_at_3": float(ndcg3),
                "ndcg_at_5": float(ndcg5),
                "best_iteration": int(model.best_iteration_ or 0),
                "n_train_queries": int(n_train_queries),
                "trained_on": "synthetic",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Sanity: a high-similarity, high-importance, recent memory must outrank a
    # low-similarity, stale, deprioritised one. If this inverts, the feature
    # order in ranker_service.py is wrong.
    strong = [[0.91, 9.0, 3.0, 3.0, 12.0, 1.0, 3.0, 0.4]]
    weak = [[0.14, 2.0, 1.0, 300.0, 0.0, 0.0, 1.0, 0.9]]
    strong_score = float(model.predict(np.asarray(strong, dtype=np.float32))[0])
    weak_score = float(model.predict(np.asarray(weak, dtype=np.float32))[0])
    print(f"\n  sanity: strong candidate {strong_score:+.3f} vs weak {weak_score:+.3f}"
          f"  -> {'✓ correct order' if strong_score > weak_score else '✗ INVERTED'}")

    print(f"\n✅ model  -> {MODEL_PATH}")
    print(f"✅ meta   -> {META_PATH}")
    print("   backend/services/ranker_service.py loads this on first use.")

    if ndcg3 < 0.70:
        print("\n⚠ NDCG@3 below the 0.70 target — raise n_estimators or N_QUERIES.")


if __name__ == "__main__":
    main()
