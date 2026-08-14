# Evaluation results

_Generated 2026-08-14 15:31 UTC · n=8 held-out founder prompts from `founder_val.jsonl` (never seen in training)._

## Persona adherence

Four mechanical checks per response, averaged. `gold` is the dataset's own answers — the target, not a model.

| Configuration | Persona score | Ends w/ question | Uses framework | No hedging | Words in band | Mean words |
|---|---|---|---|---|---|---|
| Training data (gold) | **88%** (3.50/4) | 62% | 100% | 88% | 100% | 181 |
| Base + prompt, no RAG | **81%** (3.25/4) | 100% | 100% | 88% | 38% | 150 |
| RAG + ranker (deployed) | _pending_ | — | — | — | — | — |
| Fine-tuned + RAG + ranker | _pending_ | — | — | — | — | — |

## RAG ablation

| Configuration | Memories retrieved | Precision@3 | Median latency |
|---|---|---|---|
| Training data (gold) | — | _needs labels_ | 0 ms |
| Base + prompt, no RAG | — | _needs labels_ | 1102 ms |

### Method

- **Prompts** — held-out validation split, never trained on.
- **Scoring** — four binary checks: ends with `?`; mentions a startup framework; contains no hedge phrase; 150-250 words. Deliberately mechanical, so it is reproducible and needs no judge model.
- **Precision@3** is unfilled on purpose. Counting retrieved memories is not the same as knowing they were useful, and inventing a number there would be worse than leaving it empty.

Reproduce: `python ml/eval/eval_harness.py`
