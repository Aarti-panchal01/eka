"""Text generation with three interchangeable backends.

    LLM_MODE=groq      Groq free API, base Llama 3.1 8B. Works day one.
    LLM_MODE=ollama    your merged fine-tuned personas (local, or via ngrok)
    LLM_MODE=hf_space  a HF Space (ZeroGPU) serving the merged model

Whatever the configured mode, a failure falls through to Groq, and a Groq
failure falls through to a plain apology string. The backend never 500s because
a model server went away.

# DEPLOY_LATER: set LLM_MODE=ollama once merge_lora.py has run and
# `ollama create eka-founder ...` succeeded. Nothing else needs to change.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

FALLBACK_REPLY = (
    "I'm having trouble reaching my language model right now. "
    "Your message is saved — ask me again in a moment."
)


class LLMService:
    def __init__(self) -> None:
        self._groq_client: Optional[object] = None
        self._ollama_available: Optional[bool] = None
        self._ollama_checked_at: float = 0.0
        self._ollama_models: List[str] = []
        self._last_backend: str = "none"
        self._last_error: Optional[str] = None
        # Don't re-probe a dead Ollama on every message.
        self._probe_ttl = 60.0

    # ---------------------------------------------------------- public API
    async def generate(
        self,
        prompt: str,
        mode: str = "founder",
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a reply. Returns text; never raises."""
        temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
        max_tokens = settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens

        order = self._backend_order()
        for backend in order:
            try:
                if backend == "ollama":
                    if not await self._ollama_up():
                        continue
                    text = await self._ollama(prompt, mode, system, temperature, max_tokens)
                elif backend == "hf_space":
                    text = await self._hf_space(prompt, mode, system, temperature, max_tokens)
                else:
                    text = await self._groq(prompt, system, temperature, max_tokens)

                if text and text.strip():
                    self._last_backend = backend
                    return self._postprocess(text)
                logger.warning("%s returned empty output", backend)
            except Exception as exc:
                self._last_error = f"{backend}: {exc}"
                logger.warning("LLM backend %s failed: %s", backend, exc)

        self._last_backend = "none"
        logger.error("All LLM backends failed. Last error: %s", self._last_error)
        return FALLBACK_REPLY

    def _backend_order(self) -> List[str]:
        """Configured backend first, then Groq as the universal safety net."""
        primary = settings.LLM_MODE
        if primary == "hf_space" and not settings.HF_SPACE_URL:
            logger.debug("LLM_MODE=hf_space but HF_SPACE_URL is empty")
            primary = "groq"
        order = [primary]
        if "groq" not in order:
            order.append("groq")
        return order

    # ------------------------------------------------------------- ollama
    async def _ollama_up(self) -> bool:
        """Cached liveness probe against /api/tags."""
        now = time.monotonic()
        if (
            self._ollama_available is not None
            and now - self._ollama_checked_at < self._probe_ttl
        ):
            return self._ollama_available

        self._ollama_checked_at = now
        try:
            async with httpx.AsyncClient(timeout=settings.OLLAMA_PROBE_TIMEOUT) as client:
                response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                body = response.json()
                self._ollama_models = [
                    m.get("name", "") for m in body.get("models", [])
                ]
                self._ollama_available = True
                logger.info("Ollama up — models: %s", self._ollama_models or "(none)")
            else:
                self._ollama_available = False
        except Exception as exc:
            if self._ollama_available is not False:
                logger.warning("Ollama unavailable (%s), falling back to Groq", exc)
            self._ollama_available = False
        return self._ollama_available

    async def _ollama(
        self,
        prompt: str,
        mode: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        model = settings.ollama_model_for_mode(mode)

        # If the persona model isn't registered, use any model that is rather
        # than failing — a base Llama answer beats no answer.
        if self._ollama_models and not any(
            m == model or m.startswith(f"{model}:") for m in self._ollama_models
        ):
            substitute = self._ollama_models[0]
            logger.warning(
                "Ollama model '%s' not found; using '%s'. Run "
                "ml/scripts/merge_lora.py --mode %s to create it.",
                model, substitute, mode,
            )
            model = substitute

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": settings.LLM_TOP_P,
                "top_k": settings.LLM_TOP_K,
                "num_predict": max_tokens,
                "stop": ["<|eot_id|>", "User:", "\n\nUser:"],
            },
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload
            )
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        return response.json().get("response", "")

    # --------------------------------------------------------------- groq
    @property
    def groq_client(self):
        if self._groq_client is None:
            if not settings.GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is not set")
            from groq import AsyncGroq

            self._groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        return self._groq_client

    async def _groq(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        completion = await self.groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            top_p=settings.LLM_TOP_P,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content or ""

    # ----------------------------------------------------------- hf space
    async def _hf_space(
        self,
        prompt: str,
        mode: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Talk to a Gradio/FastAPI Space. Tries the common route shapes."""
        if not settings.HF_SPACE_URL:
            raise RuntimeError("HF_SPACE_URL is not set")

        headers = {}
        if settings.HF_TOKEN:
            headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"

        attempts = [
            # A FastAPI Space written to match our own contract.
            (
                f"{settings.HF_SPACE_URL}/generate",
                {
                    "prompt": prompt,
                    "system": system or "",
                    "mode": mode,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            ),
            # A stock Gradio Space.
            (
                f"{settings.HF_SPACE_URL}/run/predict",
                {"data": [prompt, system or "", temperature, max_tokens]},
            ),
        ]

        last_error = None
        for url, payload in attempts:
            try:
                async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                    response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code} at {url}"
                    continue
                body = response.json()
                if isinstance(body, dict):
                    if "data" in body and isinstance(body["data"], list) and body["data"]:
                        return str(body["data"][0])
                    for key in ("response", "text", "output", "generated_text"):
                        if body.get(key):
                            return str(body[key])
                if isinstance(body, str):
                    return body
                last_error = f"unrecognised response shape from {url}"
            except Exception as exc:
                last_error = str(exc)
        raise RuntimeError(last_error or "hf_space unreachable")

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _postprocess(text: str) -> str:
        """Strip leaked chat-template tokens and trailing turn starts."""
        text = text.strip()
        for token in (
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
        ):
            text = text.replace(token, "")
        for marker in ("\nUser:", "\n\nUser:", "\nassistant\n"):
            index = text.find(marker)
            if index > 40:  # keep short replies that legitimately contain it
                text = text[:index]
        return text.strip()

    # ------------------------------------------------------------- health
    async def health_check(self) -> Dict[str, object]:
        ollama = await self._ollama_up()

        groq_ok = False
        if settings.GROQ_API_KEY:
            try:
                await asyncio.wait_for(
                    self._groq("Say OK.", None, 0.0, 5), timeout=15.0
                )
                groq_ok = True
            except Exception as exc:
                self._last_error = f"groq: {exc}"

        hf_space_ok = False
        if settings.HF_SPACE_URL:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(settings.HF_SPACE_URL)
                hf_space_ok = response.status_code < 500
            except Exception:
                hf_space_ok = False

        return {
            "mode": settings.LLM_MODE,
            "ollama": ollama,
            "ollama_models": self._ollama_models,
            "groq": groq_ok,
            "hf_space": hf_space_ok,
            "last_backend": self._last_backend,
            "last_error": self._last_error,
        }


llm_service = LLMService()
