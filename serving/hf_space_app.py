"""Eka's Hugging Face Space (ZeroGPU) — Option B serving endpoint.

This is the alternative to local Ollama + ngrok (see serving/merge_lora_and_
serve.py and infra/SERVING.md). Deploy this file as the entrypoint of a HF
Space with ZeroGPU hardware, and point the backend at it:

    LLM_MODE=hf_space
    HF_SPACE_URL=https://<your-username>-<space-name>.hf.space

# DEPLOY_LATER: this file only matters once at least one LoRA adapter has
# been trained and pushed to `<HF_USERNAME>/eka-<mode>-lora` on the Hub.

--------------------------------------------------------------------------
DESIGN CHOICE: one base model + four swappable PEFT adapters
--------------------------------------------------------------------------
Each of Eka's four personas (founder / chanakya / gita / reflection) has its
own LoRA adapter, but they all share the same base model
(Qwen/Qwen2.5-7B-Instruct). A merged fp16 checkpoint is ~15GB, and
ZeroGPU Spaces have limited disk + only allocate a GPU for the duration of a
single decorated call — loading four full 16GB merged models simultaneously
is not viable (64GB+ of weights, most of it identical base-model bytes
duplicated four times).

Instead this Space loads the base model ONCE and layers all four adapters on
top with PEFT's multi-adapter support:

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, ...)
    model = PeftModel.from_pretrained(base, adapter_repo("founder"), adapter_name="founder")
    model.load_adapter(adapter_repo("chanakya"), adapter_name="chanakya")
    model.load_adapter(adapter_repo("gita"), adapter_name="gita")
    model.load_adapter(adapter_repo("reflection"), adapter_name="reflection")
    ...
    model.set_adapter(request.mode)   # picks the persona per-request, cheap

Total memory is ~1 base model + 4 small LoRA deltas (a few hundred MB total
for rank-16/32 adapters), instead of 4x the base model. `mode` in the request
body selects the adapter at request time via `set_adapter`, which just swaps
which low-rank matrices are added to the frozen base weights — it does not
reload anything from disk.

ALTERNATIVE (not used here): run one Space per persona, each serving its own
fully-merged model. Tradeoff: simpler code (no adapter juggling, no risk of
adapter cross-talk), but 4x the disk/RAM footprint, 4x the cold starts to
manage, and 4x the ZeroGPU quota consumption for what is otherwise the same
underlying 8B model. Pick that route only if the adapters interact badly when
co-resident (they shouldn't — PEFT keeps them fully isolated) or if personas
need genuinely different base models later.
--------------------------------------------------------------------------

CONTRACT (must match backend/services/llm_service.py's _hf_space() exactly):
    POST /generate
        body:     {"prompt": str, "system": str, "mode": str,
                    "temperature": float, "max_tokens": int}
        response: {"response": str}
    GET /health  -> service + model status
    GET /        -> basic info
"""

import logging
import os
import threading
import time
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eka-hf-space")

# --------------------------------------------------------------------------
# ZeroGPU's `spaces` package is only installed inside an actual HF Space with
# ZeroGPU hardware. Import it optionally so this file still runs locally (or
# on a Space without ZeroGPU) — in that case @spaces.GPU becomes a no-op and
# generation just uses whatever device is available (CPU, or a real GPU if
# CUDA_VISIBLE_DEVICES is set).
# --------------------------------------------------------------------------
try:
    import spaces  # type: ignore

    GPU_AVAILABLE_VIA_ZEROGPU = True

    def gpu_decorator(duration: int = 60):
        return spaces.GPU(duration=duration)

except ImportError:
    GPU_AVAILABLE_VIA_ZEROGPU = False

    def gpu_decorator(duration: int = 60):
        """No-op fallback so `@gpu_decorator(duration=60)` works without `spaces`."""
        def _wrap(fn):
            return fn
        return _wrap


VALID_MODES = ("founder", "chanakya", "gita", "reflection")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_USERNAME = os.environ.get("HF_USERNAME", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "4096"))

# ChatML — must match ml/scripts/preprocess.py's TEMPLATE (used as a fallback
# when the tokenizer has no chat_template of its own).
FALLBACK_TEMPLATE = (
    "<|im_start|>system\n"
    "{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def adapter_repo(mode: str) -> str:
    """Matches the naming convention in ml/scripts/merge_lora.py."""
    username = HF_USERNAME or "eka"
    return f"{username}/eka-{mode}-qwen"


# --------------------------------------------------------------------------
# Lazy, lock-guarded model load. ZeroGPU only grants a GPU for the duration of
# a @spaces.GPU-decorated call, so the actual `.generate()` happens inside
# one; loading (including any device_map="auto"/.to("cuda") placement) is
# safe to do lazily on the first request because the `spaces` package patches
# CUDA init to work correctly even when called outside a currently-active
# GPU allocation.
# --------------------------------------------------------------------------
_load_lock = threading.Lock()
_model = None
_tokenizer = None
_loaded_adapters: List[str] = []
_load_error: Optional[str] = None
_quantization = "not-loaded"
_device = "not-loaded"


def _load_model_and_adapters() -> None:
    global _model, _tokenizer, _loaded_adapters, _load_error, _quantization, _device

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model %s ...", BASE_MODEL)
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    quant_config = None
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        _quantization = "4bit (bitsandbytes nf4)"
        logger.info("bitsandbytes available — loading base model in 4-bit")
    except ImportError:
        _quantization = "fp16 (bitsandbytes not installed)"
        logger.info("bitsandbytes not available — loading base model in fp16")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        quantization_config=quant_config,
        torch_dtype=torch.float16 if quant_config is None else None,
        device_map="auto",
    )
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading persona adapters for modes: %s", VALID_MODES)
    first_mode, *rest_modes = VALID_MODES
    model = PeftModel.from_pretrained(
        base, adapter_repo(first_mode), adapter_name=first_mode, token=HF_TOKEN
    )
    loaded = [first_mode]
    for mode in rest_modes:
        try:
            model.load_adapter(adapter_repo(mode), adapter_name=mode, token=HF_TOKEN)
            loaded.append(mode)
        except Exception as exc:
            logger.warning(
                "Could not load adapter for mode '%s' (%s) — that persona will "
                "fall back to '%s' until %s is trained/pushed.",
                mode, exc, first_mode, adapter_repo(mode),
            )
    model.set_adapter(first_mode)
    model.eval()

    _model = model
    _loaded_adapters = loaded
    logger.info("Model ready. Loaded adapters: %s (device=%s, quant=%s)", loaded, _device, _quantization)


def ensure_loaded() -> Optional[str]:
    """Load on first use. Returns an error string on failure, else None."""
    global _load_error
    if _model is not None:
        return None
    with _load_lock:
        if _model is not None:
            return None
        try:
            _load_model_and_adapters()
            _load_error = None
        except Exception as exc:
            logger.exception("Model load failed")
            _load_error = str(exc)
    return _load_error


def build_prompt(system: str, user: str) -> str:
    """Llama-3 chat prompt via the tokenizer's chat_template if it defines
    one, else the explicit template matching ml/scripts/preprocess.py."""
    if _tokenizer is not None and getattr(_tokenizer, "chat_template", None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        try:
            return _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:
            logger.warning("chat_template failed (%s) — using fallback template", exc)
    return FALLBACK_TEMPLATE.format(system=system, user=user)


@gpu_decorator(duration=60)
def run_generation(
    prompt: str, system: str, mode: str, temperature: float, max_tokens: int
) -> str:
    """The actual forward pass. Decorated with @spaces.GPU so ZeroGPU grants
    this call a GPU for up to `duration` seconds; on non-ZeroGPU deployments
    the decorator is a no-op and this just runs on whatever device is set."""
    import torch

    error = ensure_loaded()
    if error:
        raise RuntimeError(f"model failed to load: {error}")

    mode = mode if mode in _loaded_adapters else (_loaded_adapters[0] if _loaded_adapters else mode)
    _model.set_adapter(mode)

    text = build_prompt(system, prompt)
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    im_end = _tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [_tokenizer.eos_token_id]
    if isinstance(im_end, int) and im_end >= 0 and im_end not in eos_ids:
        eos_ids.append(im_end)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max(1, min(max_tokens, 2048)),
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            top_k=40,
            eos_token_id=eos_ids,
            pad_token_id=_tokenizer.pad_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text_out = _tokenizer.decode(generated, skip_special_tokens=True)

    # Defensive cleanup in case skip_special_tokens misses a leaked token or
    # the model echoes a new turn marker.
    for token in ("<|im_end|>", "<|im_start|>"):
        cut = text_out.find(token)
        if cut != -1:
            text_out = text_out[:cut]
    return text_out.strip()


# ------------------------------------------------------------------- FastAPI
app = FastAPI(title="Eka HF Space (ZeroGPU)", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system: str = ""
    mode: str = "founder"
    temperature: float = 0.7
    max_tokens: int = 512


class GenerateResponse(BaseModel):
    response: str


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Matches backend/services/llm_service.py's _hf_space() contract exactly:
    it posts {prompt, system, mode, temperature, max_tokens} here and reads
    body["response"]."""
    mode = request.mode if request.mode in VALID_MODES else "founder"

    import asyncio

    started = time.perf_counter()
    text = await asyncio.to_thread(
        run_generation, request.prompt, request.system, mode,
        request.temperature, request.max_tokens,
    )
    logger.info("generated %d chars in %.1fs (mode=%s)", len(text), time.perf_counter() - started, mode)
    return GenerateResponse(response=text)


@app.get("/health")
async def health() -> Dict[str, object]:
    return {
        "status": "ok" if _load_error is None else "error",
        "model_loaded": _model is not None,
        "load_error": _load_error,
        "base_model": BASE_MODEL,
        "loaded_adapters": _loaded_adapters,
        "device": _device,
        "quantization": _quantization,
        "zerogpu": GPU_AVAILABLE_VIA_ZEROGPU,
    }


@app.get("/")
async def root() -> Dict[str, object]:
    return {
        "name": "Eka HF Space (ZeroGPU)",
        "modes": list(VALID_MODES),
        "generate": "POST /generate {prompt, system, mode, temperature, max_tokens}",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))


# ==========================================================================
# HOW TO DEPLOY THIS AS A HUGGING FACE SPACE WITH ZEROGPU
# ==========================================================================
#
# --------------------------------------------------------------------
# requirements.txt  (put this file next to hf_space_app.py in the Space repo)
# --------------------------------------------------------------------
# fastapi==0.111.0
# uvicorn[standard]==0.30.0
# pydantic==2.7.1
# torch==2.3.0
# transformers==4.41.0
# peft==0.11.1
# accelerate==0.30.0
# bitsandbytes==0.43.1
# huggingface-hub==0.23.2
# spaces>=0.28.3
#
# --------------------------------------------------------------------
# README.md  (YAML frontmatter — HF Spaces read this to configure the Space;
# put it at the very top of the Space's README.md, followed by any prose)
# --------------------------------------------------------------------
# ---
# title: Eka Personas (ZeroGPU)
# emoji: 🧘
# colorFrom: indigo
# colorTo: purple
# sdk: docker
# app_port: 7860
# python_version: "3.10"
# pinned: false
# license: llama3
# short_description: Fine-tuned Llama-3 personas for Eka, served on ZeroGPU
# ---
#
# NOTE on sdk choice: ZeroGPU hardware can be requested for a `sdk: docker`
# Space as long as the `spaces` package is imported and used exactly as it is
# in this file (the `@spaces.GPU` decorator is what lets HF's scheduler grant
# this container a GPU for the duration of a call). If your account/Space
# tier only offers ZeroGPU to Gradio-SDK Spaces, use `sdk: gradio` instead and
# mount this FastAPI app inside a trivial Gradio Blocks wrapper:
#
#     import gradio as gr
#     from hf_space_app import app as fastapi_app
#     demo = gr.Blocks()
#     with demo:
#         gr.Markdown("Eka persona endpoint — see /health and /generate")
#     app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")
#
# and deploy that wrapper as the Space's app.py; /generate and /health remain
# reachable exactly as documented above because they're routes on the
# underlying FastAPI app that gr.mount_gradio_app extends rather than replaces.
#
# --------------------------------------------------------------------
# Dockerfile  (only needed for `sdk: docker`; omit entirely if using `sdk:
# gradio` — Spaces build Gradio apps without a Dockerfile)
# --------------------------------------------------------------------
# FROM python:3.10-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY hf_space_app.py .
# EXPOSE 7860
# CMD ["python", "hf_space_app.py"]
#
# --------------------------------------------------------------------
# Step by step
# --------------------------------------------------------------------
# 1. https://huggingface.co/new-space
# 2. Owner: your account. Space name: e.g. "eka-personas". SDK: Docker (or
#    Gradio — see note above). Hardware: leave as CPU basic for now; you will
#    switch to ZeroGPU in step 5 (ZeroGPU Spaces still build/idle on CPU).
# 3. Clone the new Space repo locally (git clone https://huggingface.co/
#    spaces/<you>/eka-personas), copy in hf_space_app.py, requirements.txt,
#    README.md (with the frontmatter above), and Dockerfile if using sdk:
#    docker. Commit and push.
# 4. Space Settings -> Repository secrets -> add:
#      HF_TOKEN     = a Hugging Face token with read access to your
#                      eka-<mode>-qwen adapter repos. The base model
#                      (Qwen/Qwen2.5-7B-Instruct) is ungated, so there is no
#                      license to accept — a token that can read your own
#                      private adapters is enough.
#      HF_USERNAME  = your HF username (used to build the adapter repo IDs)
#    Optional: BASE_MODEL to override the default Qwen2.5-7B-Instruct.
# 5. Space Settings -> Hardware -> select "ZeroGPU". This requires a Pro
#    account or a Space that has been granted ZeroGPU access (HF offers a
#    limited free ZeroGPU quota via Spaces run on community grants — check
#    the Hardware tab for current eligibility).
# 6. Wait for the build to finish, then GET https://<you>-eka-personas.
#    hf.space/health — model_loaded will be false until the first /generate
#    call triggers the lazy load (this is intentional: it keeps the Space's
#    idle state cheap and defers the ~20-40s model+adapter load to the first
#    real request instead of the container's startup).
# 7. Point the backend at it:
#      LLM_MODE=hf_space
#      HF_SPACE_URL=https://<you>-eka-personas.hf.space
# ==========================================================================
