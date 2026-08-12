# Running the Kaggle training sessions

Six training scripts live in `training/`. Four are the persona QLoRA runs and
are the reason this doc exists; two are short auxiliary runs you can do in a
single sitting.

| # | Script | Produces | T4 time |
|---|---|---|---|
| 1 | `train_founder_lora_kaggle.py` | `eka-founder-lora` | ~2.5–3.5 h |
| 2 | `train_chanakya_lora_kaggle.py` | `eka-chanakya-lora` | ~2.5–3.5 h |
| 3 | `train_gita_lora_kaggle.py` | `eka-gita-lora` | ~2.5–3.5 h |
| 4 | `train_reflection_lora_kaggle.py` | `eka-reflection-lora` | ~2.5–3.5 h |
| 5 | `train_embeddings_kaggle.py` | `eka-embeddings` | ~20–40 min |
| 6 | `train_classifiers_kaggle.py` | `eka-complexity`, `eka-sentiment` | ~20–40 min |

Each script is fully self-contained — it imports nothing from the Eka project.
All nine target HF repos already exist and are private.

**Prefer the notebooks in `ml/notebooks/`** for runs 1–4: Kaggle → New Notebook
→ File → Import Notebook. They are generated from these same scripts by
`scripts/build_kaggle_notebooks.py`, split one cell per section, so a failure
names the stage it died in and you can re-run just the push-to-Hub cell if a
3-hour train succeeds and the upload 401s. Pasting the whole `.py` into one
cell is still fine — same code — you just lose both of those. Runs 5 and 6
have no notebook yet; paste those.

**Run 5 cannot work yet — its dataset does not exist.** See "Session order"
below before you spend a GPU slot on it.

---

## Prerequisite — the datasets must be on the Hub first

Every script reads from `amijackofalltrades/eka-datasets`. Nothing will train
until that repo has content. Locally:

```bash
python ml/scripts/preprocess.py       # -> ml/data/splits/{mode}_{train,val}.jsonl
python ml/scripts/upload_to_hf.py     # -> pushes splits + aux datasets to the Hub
```

Confirm before you open Kaggle — a missing split fails ~10 minutes into a run,
after the model download, which is a slow way to learn about a typo:

```bash
python -c "
from huggingface_hub import HfApi; import os
from dotenv import load_dotenv; load_dotenv('.env')
files = HfApi(token=os.getenv('HF_TOKEN')).list_repo_files(
    'amijackofalltrades/eka-datasets', repo_type='dataset')
print('\n'.join(sorted(files)))"
```

You want, at minimum, `founder_train.jsonl` and `founder_val.jsonl` before
Session 1. The other personas can land later, as long as each is up before its
own session.

---

## One-time Kaggle account setup

Do this once. It applies to every notebook afterwards.

1. **Add-ons → Secrets**, add all three:
   - `HF_TOKEN` — must have **write** permission
   - `HF_USERNAME` — `amijackofalltrades`
   - `WANDB_API_KEY`
2. **Tick the checkbox next to each secret** to attach it to the notebook.
   An added-but-unattached secret reads back as an empty string; the auth cell
   catches this and exits with a clear message, but only after you have waited
   for the notebook to boot.
3. **Accept the Llama 3 license** at
   [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
   using the **same account** `HF_TOKEN` belongs to (`amijackofalltrades`).
   Skip this and the base-model download 403s. Approval is usually instant but
   is occasionally queued — do it before Session 1, not during.
4. Note your **GPU quota: 30 hours/week**, resetting Saturday 00:00 UTC. Four
   persona runs at ~3.5 h worst case is ~14 h, so all four fit in one week with
   room to retry a failure. The two short runs fit easily too.

---

## Per-session procedure

Identical for all six. Only the pasted script changes.

1. **kaggle.com/code → New Notebook**.
2. Name it for the run (`eka-founder-lora`) so you can find it later.
3. Right-hand **Settings** panel:
   - **Accelerator** → `GPU T4 x2`. The script uses one GPU; x2 is selected
     because it also raises the RAM ceiling.
   - **Internet** → `On`. Required — it downloads the base model and pushes
     results.
   - **Persistence** → `Files only` if offered. This is what lets a restarted
     session find its checkpoints.
4. Delete the default cell. Paste the **entire** script file into one cell.
5. **Save Version → Save & Run All (Commit)**.

   **This is the step that matters.** "Save & Run All (Commit)" is Kaggle's
   background execution — the run continues after you close the tab. If you
   instead hit the interactive **Run All** button and close the browser, the
   session is killed and the run dies. Interactive sessions also idle out after
   ~20–40 minutes of no interaction, which a 3-hour run will not survive.

6. Close the tab. Check back via **kaggle.com/code → Your Work → the
   notebook → Logs**.
7. When it finishes, confirm the adapter actually landed on the Hub:

   ```bash
   python -c "
   from huggingface_hub import HfApi; import os
   from dotenv import load_dotenv; load_dotenv('.env')
   print(HfApi(token=os.getenv('HF_TOKEN')).list_repo_files(
       'amijackofalltrades/eka-founder-lora'))"
   ```

   Expect `adapter_model.safetensors` and `adapter_config.json`. A repo with
   only a `README.md` means the run failed before the push — read the logs.

---

## Session order

Nothing depends on anything else, so the order is only about de-risking. Run
**Session 1 (founder) first and watch it to completion**, because every
mistake you are going to make — unattached secret, unaccepted license, missing
split — surfaces there. Sessions 2–4 are then the same run with a different
`MODE`.

- **Session 1 — founder.** Paste `train_founder_lora_kaggle.py`. GPU T4 x2,
  Internet on, Save & Run All (Commit). Verify the adapter on the Hub before
  starting anything else.
- **Session 2 — chanakya.** Same day is fine; you have the weekly quota for it.
- **Session 3 — gita.**
- **Session 4 — reflection.**
- **Session 6 — classifiers.** Short, and its data is ready: 
  `complexity_labeled.jsonl` exists and the sentiment half pulls `go_emotions`
  straight from the Hub. Run whenever convenient; needs no persona adapter.

- **Session 5 — embeddings. BLOCKED — check this before you open Kaggle.**
  `train_embeddings_kaggle.py` downloads `embedding_triplets.jsonl` from the
  dataset repo, and **that file has never been generated**. It is not in
  `ml/datasets/`, so `upload_to_hf.py` skips it (printing `⏭ not found`) and
  the session fails after the base-model download. Build it first:

  ```bash
  python ml/scripts/generate_embedding_triplets.py     # default --target 6000
  python ml/scripts/upload_to_hf.py                    # re-push the repo
  ```

  Two things to know before starting it. It is **Groq-only**: it calls
  `get_groq_client()` directly rather than going through the provider rotator,
  so the four Mistral keys cannot help, and Groq's ~100k tokens/day is the
  binding limit — 6,000 paraphrase calls will not fit in one day. It is
  resume-safe, so run it across days or pass a smaller `--target`.

  Nothing else depends on it. Sessions 1–4 and 6 are unaffected.

You can have two Kaggle sessions running concurrently, so 1+2 and 3+4 can
overlap. Do not overlap on the first attempt — debug serially, then parallelise.

---

## If a session dies partway

The four persona scripts call `find_latest_checkpoint(OUTPUT_DIR)` and pass the
result to `resume_from_checkpoint`, so a restart picks up from the last saved
checkpoint rather than step 0 — provided the checkpoints in
`/kaggle/working/{mode}_lora` survived. Re-run the same notebook version; the
startup log prints which checkpoint it resumed from. If it prints `None` and
you expected a resume, the working directory was wiped and you are starting
over.

**Note the asymmetry:** `train_embeddings_kaggle.py` and
`train_classifiers_kaggle.py` have **no** checkpoint-resume logic — a failure
means re-running from the start. That is a deliberate tradeoff given they take
20–40 minutes, not hours; it is not an oversight to fix under time pressure.

Common failures, in the order you are likely to hit them:

| Symptom | Cause |
|---|---|
| Auth cell exits immediately | Secret added but not attached (step 2 above) |
| `403` on base model download | Llama 3 license not accepted by `amijackofalltrades` |
| `EntryNotFoundError` on a `.jsonl` | `upload_to_hf.py` has not been run for that persona |
| CUDA OOM | Accelerator is T4 **x1**; switch to x2, or lower `MAX_SEQ_LEN` |
| Run vanished on tab close | Interactive Run All instead of Save & Run All (Commit) |

---

## After all four adapters exist

`ml/scripts/merge_lora.py` merges an adapter into the base weights, and
`serving/merge_lora_and_serve.py` serves the merged model. `infra/SERVING.md`
covers where those can actually run — note that none of the merged 8B models
fit Render's free tier.
