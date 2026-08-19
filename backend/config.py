"""Eka configuration — single source of truth for every environment variable.

Everything is loaded from the project-root .env file (or the real process
environment, which is what Render/Docker will provide). Import the singleton:

    from config import settings
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/config.py -> backend/ -> eka/
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
PROMPTS_DIR = BACKEND_DIR / "prompts"
ML_DIR = PROJECT_ROOT / "ml"


class Settings(BaseSettings):
    """All Eka runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------- app
    APP_NAME: str = "Eka"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["local", "docker", "render", "production"] = "local"
    DEBUG: bool = True
    SECRET_KEY: str = "change-me-generate-64-random-chars"
    FRONTEND_URL: str = "http://localhost:5173"
    CORS_ORIGINS: str = "*"

    # ---------------------------------------------------------------- LLM
    # "groq"     -> Groq free API, base Llama. Works immediately, no infra.
    # "ollama"   -> local/ngrok Ollama serving the merged fine-tuned personas.
    # "hf_space" -> Hugging Face Space (ZeroGPU) serving the merged model.
    LLM_MODE: Literal["groq", "ollama", "hf_space"] = "groq"

    GROQ_API_KEY: str = ""
    # Chat/inference model (fast + free). Used for every user-facing generation
    # while LLM_MODE=groq.
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    # Bigger model used only by the offline data-generation scripts.
    # NOTE: llama-3.1-70b-versatile was decommissioned by Groq; 3.3 is the
    # current 70B. Override with GROQ_GEN_MODEL if Groq rotates models again.
    GROQ_GEN_MODEL: str = "llama-3.3-70b-versatile"

    # DEPLOY_LATER: swap LLM_MODE=ollama once Ollama is reachable (local, or
    # exposed with `ngrok http 11434` and OLLAMA_BASE_URL set to the tunnel).
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_PREFIX: str = "eka"  # -> eka-founder, eka-chanakya, ...

    # DEPLOY_LATER: HF Space (ZeroGPU) endpoint serving the merged model.
    HF_SPACE_URL: str = ""

    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    LLM_TOP_K: int = 40
    LLM_MAX_TOKENS: int = 512
    LLM_TIMEOUT_SECONDS: float = 60.0
    OLLAMA_PROBE_TIMEOUT: float = 10.0

    # ---------------------------------------------------- Hugging Face Hub
    HF_TOKEN: str = ""
    # Default matches the account HF_TOKEN authenticates as; .env overrides.
    HF_USERNAME: str = "amijackofalltrades"
    # The old api-inference.huggingface.co host is retired — it no longer even
    # resolves in DNS. Everything now goes through the router. Note the token
    # needs the "Make calls to Inference Providers" permission or every request
    # 403s with "This authentication method does not have sufficient
    # permissions"; regenerate the token at huggingface.co/settings/tokens.
    HF_INFERENCE_BASE: str = "https://router.huggingface.co/hf-inference/models"

    # Model repo names (joined with HF_USERNAME via the properties below).
    EMBEDDING_REPO: str = "eka-embeddings"
    COMPLEXITY_REPO: str = "eka-complexity"
    SENTIMENT_REPO: str = "eka-sentiment"
    SUMMARIZER_REPO: str = "eka-summarizer"
    DATASET_REPO: str = "eka-datasets"
    # Base models used as fallback until the fine-tunes exist on the Hub.
    EMBEDDING_FALLBACK_MODEL: str = "BAAI/bge-base-en-v1.5"
    SUMMARIZER_FALLBACK_MODEL: str = "t5-small"
    SENTIMENT_FALLBACK_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"
    TOXICITY_MODEL: str = "unitary/toxic-bert"

    # ----------------------------------------------------------- database
    DATABASE_URL: str = "postgresql+asyncpg://eka:eka@localhost:5432/eka"
    DB_ECHO: bool = False
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 5
    DB_POOL_RECYCLE: int = 1800

    # ---------------------------------------------------------- vector db
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "memories"
    EMBEDDING_DIM: int = 768
    QDRANT_TIMEOUT: float = 30.0

    # -------------------------------------------------------------- voice
    SARVAM_API_KEY: str = ""
    SARVAM_TTS_URL: str = "https://api.sarvam.ai/text-to-speech"
    SARVAM_STT_URL: str = "https://api.sarvam.ai/speech-to-text"
    SARVAM_TTS_MODEL: str = "bulbul:v2"
    # saarika:v2 is deprecated — the API returns
    # "Model 'saarika:v2' has been deprecated. Please use 'saarika:v2.5'".
    SARVAM_STT_MODEL: str = "saarika:v2.5"
    SARVAM_LANGUAGE: str = "en-IN"
    SARVAM_SAMPLE_RATE: int = 22050

    # Every speaker below was verified against the live Sarvam API — each one
    # returns real WAV audio. All four are env-overridable so you can swap
    # voices without touching code.
    #
    # The full valid speaker list, straight from the API's own 400 response
    # (send a bogus `speaker` and it enumerates them — better than the docs,
    # which drift):
    #   anushka abhilash manisha vidya arya karun hitesh aditya ritu priya
    #   neha rahul pooja rohan simran kavya amit dev ishita shreya ratan
    #   varun manan sumit roopa kabir aayan shubh ashutosh advait anand
    #   tanya tarun sunny mani gokul vijay shruti suhani mohit kavitha
    #   rehan soham rupali
    #
    # NOTE: the older v1-era names (arvind, amol, maya, meera, pavithra...) are
    # gone — using them returns 400, not a fallback voice.
    SARVAM_VOICE_FOUNDER: str = "karun"       # male, firm — the blunt operator
    SARVAM_VOICE_CHANAKYA: str = "abhilash"   # male, measured — the strategist
    SARVAM_VOICE_GITA: str = "anushka"        # female, serene — the guide
    SARVAM_VOICE_REFLECTION: str = "vidya"    # female, calm — the therapist
    TTS_CHUNK_CHARS: int = 500
    TTS_CACHE_SIZE: int = 100

    # ------------------------------------------------- classifier serving
    # Empty string = "run the model in-process instead of calling a service".
    # On Render free tier we keep these empty and lazy-load in the same process.
    COMPLEXITY_SERVE_URL: str = ""
    RANKER_SERVE_URL: str = ""
    SENTIMENT_SERVE_URL: str = ""
    SERVICE_TIMEOUT: float = 8.0
    # Set false on 512MB Render instances if memory becomes the binding limit;
    # the heuristic fallbacks keep everything working.
    ENABLE_LOCAL_CLASSIFIERS: bool = True
    RANKER_MODEL_PATH: str = str(ML_DIR / "models" / "ranker" / "eka_ranker.txt")

    # ------------------------------------------------------------- tuning
    # 80, down from 150. At 150 almost nothing in a real conversation
    # qualified — "my company is called Inverix and we sell to schools" is 60
    # characters and is exactly the kind of fact worth keeping, so the
    # Knowledge page stayed empty while chat history filled up. The
    # first-person and question-ratio checks in _worth_remembering still do the
    # real filtering; this length floor is only meant to skip one-liners like
    # "ok" and "thanks". Raise it again if the store gets noisy.
    AUTO_MEMORY_MIN_CHARS: int = 80
    MAX_TAGS: int = 5
    EMBED_CACHE_SIZE: int = 500
    HF_COLD_START_RETRIES: int = 3
    HF_COLD_START_WAIT: int = 20

    # ------------------------------------------------------------- wandb
    WANDB_API_KEY: str = ""
    WANDB_PROJECT: str = "eka"

    # --------------------------------------------------------- validators
    @field_validator("DATABASE_URL")
    @classmethod
    def _force_async_driver(cls, v: str) -> str:
        """Supabase hands out `postgresql://...`; SQLAlchemy async needs asyncpg."""
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql://", 1)
        if v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @field_validator("QDRANT_URL", "OLLAMA_BASE_URL", "HF_SPACE_URL")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    # --------------------------------------------------------- properties
    @property
    def cors_origin_list(self) -> List[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        origins = [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]
        if self.FRONTEND_URL and self.FRONTEND_URL not in origins:
            origins.append(self.FRONTEND_URL)
        return origins

    def hub_repo(self, repo: str) -> str:
        """`eka-embeddings` -> `amijackofalltrades/eka-embeddings`."""
        if "/" in repo:
            return repo
        if not self.HF_USERNAME:
            return repo
        return f"{self.HF_USERNAME}/{repo}"

    @property
    def embedding_model_id(self) -> str:
        return self.hub_repo(self.EMBEDDING_REPO)

    @property
    def complexity_model_id(self) -> str:
        return self.hub_repo(self.COMPLEXITY_REPO)

    @property
    def sentiment_model_id(self) -> str:
        return self.hub_repo(self.SENTIMENT_REPO)

    @property
    def summarizer_model_id(self) -> str:
        return self.hub_repo(self.SUMMARIZER_REPO)

    @property
    def dataset_repo_id(self) -> str:
        return self.hub_repo(self.DATASET_REPO)

    def voice_for_mode(self, mode: str) -> str:
        return {
            "founder": self.SARVAM_VOICE_FOUNDER,
            "chanakya": self.SARVAM_VOICE_CHANAKYA,
            "gita": self.SARVAM_VOICE_GITA,
            "reflection": self.SARVAM_VOICE_REFLECTION,
        }.get(mode, self.SARVAM_VOICE_REFLECTION)

    def ollama_model_for_mode(self, mode: str) -> str:
        return f"{self.OLLAMA_MODEL_PREFIX}-{mode}"

    def prompt_path(self, mode: str) -> Path:
        return PROMPTS_DIR / f"{mode}.txt"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

VALID_MODES = ("founder", "chanakya", "gita", "reflection")
COMPLEXITY_LABELS = ("simple", "normal", "complex", "deep")
SENTIMENT_LABELS = (
    "positive",
    "neutral",
    "negative",
    "reflective",
    "anxious",
    "motivated",
)
