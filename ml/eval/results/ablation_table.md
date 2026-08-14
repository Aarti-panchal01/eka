# Evaluation results

_Generated 2026-08-14 15:44 UTC · n=50 held-out founder prompts from `founder_val.jsonl` (never seen in training)._

## Persona adherence

Four mechanical checks per response, averaged. `gold` is the dataset's own answers — the target, not a model.

| Configuration | Persona score | Ends w/ question | Uses framework | No hedging | Words in band | Mean words |
|---|---|---|---|---|---|---|
| Training data (gold) | **94%** (3.74/4) | 78% | 100% | 96% | 100% | 189 |
| Base + prompt, no RAG | **93%** (3.73/4) | 100% | 100% | 100% | 73% | 159 |
| RAG + ranker (deployed) | **78%** (3.14/4) | 90% | 82% | 82% | 60% | 215 |
| Fine-tuned + RAG + ranker | _pending_ | — | — | — | — | — |

## RAG ablation

| Configuration | Memories retrieved | Precision@3 | Median latency |
|---|---|---|---|
| Training data (gold) | — | _needs labels_ | 0 ms |
| Base + prompt, no RAG | — | _needs labels_ | 1168 ms |
| RAG + ranker (deployed) | 4.7 | _needs labels_ | 16796 ms |

### Method

- **Prompts** — held-out validation split, never trained on.
- **Scoring** — four binary checks: ends with `?`; mentions a startup framework; contains no hedge phrase; 150-250 words. Deliberately mechanical, so it is reproducible and needs no judge model.
- **Precision@3** is unfilled on purpose. Counting retrieved memories is not the same as knowing they were useful, and inventing a number there would be worse than leaving it empty.

Reproduce: `python ml/eval/eval_harness.py`
