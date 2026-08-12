# ============================================================================
# EKA — Makefile
#
# On Windows this needs a `make` that isn't cmd.exe's: use Git Bash (ships
# with Git for Windows), WSL, or `choco install make`. Every recipe below is
# a single, plain command — if you don't have `make` at all, just copy the
# command out of the target and run it by hand in your normal shell.
#
#   make            -> shows this help (default goal)
#   make help
# ============================================================================

.DEFAULT_GOAL := help

PY   ?= python
BASE ?= http://localhost:8000

.PHONY: help install install-ml \
        data-founder data-chanakya data-gita data-reflection data-all \
        data-complexity data-triplets preprocess upload-hf \
        train-founder train-chanakya train-gita train-reflection \
        train-embeddings train-classifiers train-ranker \
        merge-founder merge-all \
        serve-local serve-classifiers docker-up docker-down \
        migrate migrate-new test status syntax-check clean deploy-render

help: ## Show this help (this list is generated from the comments below)
	@echo "EKA — available make targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Override defaults: make serve-local PY=python3 / make status BASE=https://eka.onrender.com"

# ============================================================== setup
install: ## Install backend runtime deps (pip install -r backend/requirements.txt)
	$(PY) -m pip install -r backend/requirements.txt

install-ml: ## Install ML data-gen + training deps (pip install -r ml/requirements.txt)
	$(PY) -m pip install -r ml/requirements.txt

# ================================================================ data
data-founder: ## Generate the founder persona dataset (Groq API, ~45min)
	$(PY) ml/scripts/generate_founder_data.py

data-chanakya: ## Generate the chanakya persona dataset (Groq API)
	$(PY) ml/scripts/generate_chanakya_data.py

data-gita: ## Generate the gita persona dataset (Groq API)
	$(PY) ml/scripts/generate_gita_data.py

data-reflection: ## Generate the reflection persona dataset (Groq API)
	$(PY) ml/scripts/generate_reflection_data.py

data-all: ## Run all four persona data-gen scripts, in order
	$(PY) ml/scripts/generate_founder_data.py
	$(PY) ml/scripts/generate_chanakya_data.py
	$(PY) ml/scripts/generate_gita_data.py
	$(PY) ml/scripts/generate_reflection_data.py

data-complexity: ## Generate the complexity-router dataset (templates, no API)
	$(PY) ml/scripts/generate_complexity_data.py

data-triplets: ## Build embedding triplets from the persona datasets (Groq API)
	$(PY) ml/scripts/generate_embedding_triplets.py

preprocess: ## Turn persona datasets into Llama-3 chat-formatted train/val splits
	$(PY) ml/scripts/preprocess.py

upload-hf: ## Push data/splits to the HF Hub dataset repo (needs HF_TOKEN + HF_USERNAME)
	$(PY) ml/scripts/upload_to_hf.py

# ============================================================ training
# The four persona LoRAs, the embedding fine-tune, and the classifiers all
# train on a Kaggle T4 (free), not on this machine — Llama-3-8B QLoRA and a
# transformers fine-tune need a GPU this laptop doesn't have. These targets
# deliberately do NOT invoke python locally; they print what to do instead.

train-founder: ## Print Kaggle upload instructions for the founder LoRA (does NOT train locally)
	@echo "This trains on KAGGLE, not locally — Llama-3-8B QLoRA needs a GPU."
	@echo "  1. https://kaggle.com/code -> New Notebook -> GPU T4 x2 accelerator"
	@echo "  2. File > Upload notebook -> training/train_founder_lora_kaggle.py"
	@echo "  3. Add-ons > Secrets -> add HF_TOKEN, HF_USERNAME, WANDB_API_KEY"
	@echo "  4. Save & Run All (Commit) -> enables background execution"
	@echo "  5. Output lands at https://huggingface.co/<HF_USERNAME>/eka-founder-lora"

train-chanakya: ## Print Kaggle upload instructions for the chanakya LoRA (does NOT train locally)
	@echo "This trains on KAGGLE, not locally — Llama-3-8B QLoRA needs a GPU."
	@echo "  1. https://kaggle.com/code -> New Notebook -> GPU T4 x2 accelerator"
	@echo "  2. File > Upload notebook -> training/train_chanakya_lora_kaggle.py"
	@echo "  3. Add-ons > Secrets -> add HF_TOKEN, HF_USERNAME, WANDB_API_KEY"
	@echo "  4. Save & Run All (Commit) -> enables background execution"
	@echo "  5. Output lands at https://huggingface.co/<HF_USERNAME>/eka-chanakya-lora"

train-gita: ## Print Kaggle upload instructions for the gita LoRA (does NOT train locally)
	@echo "This trains on KAGGLE, not locally — Llama-3-8B QLoRA needs a GPU."
	@echo "  1. https://kaggle.com/code -> New Notebook -> GPU T4 x2 accelerator"
	@echo "  2. File > Upload notebook -> training/train_gita_lora_kaggle.py"
	@echo "  3. Add-ons > Secrets -> add HF_TOKEN, HF_USERNAME, WANDB_API_KEY"
	@echo "  4. Save & Run All (Commit) -> enables background execution"
	@echo "  5. Output lands at https://huggingface.co/<HF_USERNAME>/eka-gita-lora"

train-reflection: ## Print Kaggle upload instructions for the reflection LoRA (does NOT train locally)
	@echo "This trains on KAGGLE, not locally — Llama-3-8B QLoRA needs a GPU."
	@echo "  1. https://kaggle.com/code -> New Notebook -> GPU T4 x2 accelerator"
	@echo "  2. File > Upload notebook -> training/train_reflection_lora_kaggle.py"
	@echo "  3. Add-ons > Secrets -> add HF_TOKEN, HF_USERNAME, WANDB_API_KEY"
	@echo "  4. Save & Run All (Commit) -> enables background execution"
	@echo "  5. Output lands at https://huggingface.co/<HF_USERNAME>/eka-reflection-lora"

train-embeddings: ## Print Kaggle upload instructions for the embedding fine-tune (does NOT train locally)
	@echo "This trains on KAGGLE, not locally — needs a GPU for a reasonable runtime."
	@echo "  1. https://kaggle.com/code -> New Notebook -> GPU T4 x2 accelerator"
	@echo "  2. File > Upload notebook -> training/train_embeddings_kaggle.py"
	@echo "  3. Add-ons > Secrets -> add HF_TOKEN, HF_USERNAME, WANDB_API_KEY"
	@echo "  4. Save & Run All (Commit) -> enables background execution"
	@echo "  5. Output lands at https://huggingface.co/<HF_USERNAME>/eka-embeddings"

train-classifiers: ## Print Kaggle upload instructions for complexity+sentiment+summarizer (does NOT train locally)
	@echo "This trains on KAGGLE, not locally — three transformer fine-tunes in one notebook."
	@echo "  1. https://kaggle.com/code -> New Notebook -> GPU T4 x2 accelerator"
	@echo "  2. File > Upload notebook -> training/train_classifiers_kaggle.py"
	@echo "  3. Add-ons > Secrets -> add HF_TOKEN, HF_USERNAME, WANDB_API_KEY"
	@echo "  4. Save & Run All (Commit) -> enables background execution"
	@echo "  5. Output lands at eka-complexity, eka-sentiment, eka-summarizer on the Hub"

train-ranker: ## Train the LightGBM memory reranker LOCALLY (CPU, <10 min, no Kaggle)
	$(PY) training/train_ranker_local.py

# ================================================================ merge
merge-founder: ## Merge the founder LoRA into base Llama-3 + prep an Ollama Modelfile
	$(PY) ml/scripts/merge_lora.py --mode founder

merge-all: ## Merge all four persona LoRAs into base Llama-3 (~16GB RAM/disk each)
	$(PY) ml/scripts/merge_lora.py --all

# ================================================================ serve
serve-local: ## Run the FastAPI backend locally with auto-reload
	cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

serve-classifiers: ## Run the three standalone classifier services in the background (optional)
	@echo "Starting complexity(:8004) / ranker(:8005) / sentiment(:8007) in the background..."
	@echo "NOTE: the backend loads these in-process by default (ENABLE_LOCAL_CLASSIFIERS=true) —"
	@echo "you only need this if you're splitting services across machines/processes. If these"
	@echo "background jobs don't suit your shell (cmd.exe has no '&'), just open three terminals"
	@echo "and run each 'python serving/*.py' by hand, or use: docker compose --profile classifiers up"
	$(PY) serving/complexity_serve.py & \
	$(PY) serving/ranker_serve.py & \
	$(PY) serving/sentiment_serve.py &

# =============================================================== docker
docker-up: ## Start postgres + qdrant + backend via docker compose
	docker compose up -d

docker-down: ## Stop the docker compose stack
	docker compose down

# =============================================================== database
migrate: ## Apply all pending Alembic migrations (backend/alembic)
	cd backend && alembic upgrade head

migrate-new: ## Autogenerate a new migration — usage: make migrate-new MSG="add table"
	cd backend && alembic revision --autogenerate -m "$(MSG)"

# ================================================================= quality
test: ## Run the 7 end-to-end smoke tests against a LIVE backend (make serve-local first)
	EKA_BASE_URL=$(BASE) $(PY) tests/test_e2e.py

status: ## Run the build-status dashboard — what's trained, what's live, what's missing
	$(PY) scripts/check_build_status.py

syntax-check: ## ast.parse every .py file in the repo and report any that fail
	@failed=0; total=0; \
	for f in $$(find . -name "*.py" \
		-not -path "*/__pycache__/*" -not -path "*/.git/*" \
		-not -path "*/node_modules/*" -not -path "*/venv/*" -not -path "*/.venv/*"); do \
		total=$$((total+1)); \
		err=$$($(PY) -c "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8', errors='replace').read())" "$$f" 2>&1); \
		if [ $$? -ne 0 ]; then \
			failed=$$((failed+1)); \
			echo "FAIL $$f"; \
			echo "$$err" | sed 's/^/    /'; \
		fi; \
	done; \
	echo ""; \
	echo "checked $$total files, $$failed failed"

clean: ## Remove __pycache__ / .pytest_cache (NEVER touches ml/datasets or ml/models — those are your trained work)
	@echo "Removing __pycache__ and .pytest_cache only. Datasets and trained models are never deleted here."
	find . -type d -name "__pycache__" -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true

deploy-render: ## Print the steps to ship the backend to Render
	@echo "Render deploys straight from git — there is no separate build/upload step here."
	@echo "  1. git add -A && git commit -m \"your message\""
	@echo "  2. git push origin main"
	@echo "  3. https://dashboard.render.com -> eka-backend -> watch the deploy log"
	@echo "     (first deploy: dashboard.render.com/blueprints -> New Blueprint Instance -> this repo)"
	@echo "  4. Once live, confirm: curl $(BASE)/health"
	@echo "  5. Set up infra/uptimerobot_note.txt's monitor so the free instance doesn't sleep"
