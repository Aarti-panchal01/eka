# Eka

A lifelong AI companion with four trained personas, semantic memory, dynamic
retrieval, and voice.

Eka is not a chatbot with a system prompt. It is four fine-tuned personas over
a memory that grows for years, with retrieval depth that scales to how hard the
question actually is.

## The four modes

| Mode | Voice | What it does |
|---|---|---|
| **founder** | brutally honest entrepreneur | unit economics, PMF signals, names the problem you're avoiding |
| **chanakya** | strategic advisor (Arthashastra) | power dynamics, timing, leverage — never moralizes |
| **gita** | spiritual guide (Krishna/Arjuna) | dharma, detachment from results, cited verses |
| **reflection** | CBT/ACT therapist | never advises, only reflects and asks |

## Architecture

```
React (Vercel)
      │
      ▼
FastAPI backend (Render free tier)
      │
      ├── rag_service ......... 15-step pipeline, the brain of every message
      │     ├── complexity ..... DistilBERT 4-class -> how much context to pull
      │     ├── Qdrant Cloud ... 768-dim semantic memory search
      │     ├── LightGBM ....... reranks the candidate pool (8 features)
      │     └── llm_service .... Groq | Ollama (fine-tuned) | HF Space
      ├── memory_service ...... Postgres (Supabase) + Qdrant kept in sync
      ├── tts / asr ........... Sarvam Bulbul V4 + Saarika V2
      └── insight_service ..... daily summary, mood trend, goal alignment
```

Every external call has a graceful fallback. If Qdrant, the ranker, the
classifiers, Sarvam, and Ollama are all down, Eka still answers — just with
less context and no voice.

## Retrieval depth scales with complexity

| complexity | memories | history | candidate pool |
|---|---|---|---|
| simple | 1 | 2 | 10 |
| normal | 2 | 3 | 15 |
| complex | 3 | 5 | 25 |
| deep | 5 | 7 | 40 |

## Trained models (all on HF Hub, all free to train on Kaggle T4)

| Repo | Base | Task |
|---|---|---|
| `eka-founder-qwen` | Qwen2.5-7B-Instruct | QLoRA persona |
| `eka-chanakya-qwen` | Qwen2.5-7B-Instruct | QLoRA persona |
| `eka-gita-qwen` | Qwen2.5-7B-Instruct | QLoRA persona |
| `eka-reflection-qwen` | Qwen2.5-7B-Instruct | QLoRA persona |
| `eka-embeddings` | BGE-base-en-v1.5 | triplet fine-tune |
| `eka-complexity` | DistilBERT | 4-class routing |
| `eka-sentiment` | DistilRoBERTa | 6-class emotion |
| `eka-summarizer` | T5-small | daily summaries |
| ranker (local) | LightGBM | memory reranking |

## Quickstart

```bash
cp .env.example .env          # fill in keys
pip install -r backend/requirements.txt

# 1. generate training data (CPU only, Groq API)
make data-founder             # ...and data-chanakya / data-gita / data-reflection
make data-complexity          # no API needed
make data-triplets

# 2. ship data to the Hub, then train on Kaggle
make preprocess && make upload-hf
#   -> upload training/train_founder_lora_kaggle.py as a Kaggle notebook,
#      add HF_TOKEN / WANDB_API_KEY / HF_USERNAME as secrets,
#      enable background execution, run.

make train-ranker             # CPU, <10 minutes, no Kaggle needed

# 3. run it
make serve-local              # http://localhost:8000/docs
make status                   # what's trained, what's live, what's missing
make test                     # 7 end-to-end tests
```

## Deployment

- **Backend** → Render free tier (`render.yaml` is committed). Keep it awake
  with an UptimeRobot monitor on `/health` — see `infra/uptimerobot_note.txt`.
- **Frontend** → Vercel, `VITE_EKA_API_URL` pointed at the Render URL.
- **Fine-tuned LLM** → either `ngrok http 11434` over local Ollama, or a HF
  Space with ZeroGPU. Until then `LLM_MODE=groq` serves base Llama and
  everything else works unchanged.

Total monthly cost: ₹0.

## Layout

```
backend/    FastAPI app: routes, services, models, persona prompts
ml/         data generation scripts + generated datasets
training/   self-contained Kaggle notebooks (no project imports)
serving/    optional standalone classifier microservices
infra/      deployment notes and helper scripts
tests/      end-to-end integration tests
scripts/    build status dashboard
```
