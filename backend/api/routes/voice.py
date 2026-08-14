"""Voice routes — speech to text, text to speech."""

import logging

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from models.schemas import STTResponse, TTSRequest
from services.asr_service import asr_service
from services.tts_service import tts_service

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_AUDIO_BYTES = 25 * 1024 * 1024


@router.post("/stt", response_model=STTResponse)
async def speech_to_text(
    file: UploadFile = File(...),
    # Form field rather than JSON: this endpoint is multipart. Optional so
    # existing callers keep working on the configured default.
    language: str = Form("en-IN"),
) -> STTResponse:
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio too large (max 25MB)")

    text = await asr_service.transcribe(
        audio, file.filename or "audio.wav", language=language
    )
    return STTResponse(text=text, backend=asr_service._last_backend)


@router.post(
    "/tts",
    responses={200: {"content": {"audio/wav": {}}}, 503: {"description": "TTS unavailable"}},
)
async def text_to_speech(payload: TTSRequest) -> Response:
    audio = await tts_service.synthesize(
        payload.text, payload.mode, language=payload.language
    )
    if not audio:
        # 503 rather than 500: voice is an enhancement, and the client is
        # expected to carry on showing the text reply.
        raise HTTPException(
            status_code=503,
            detail="Text-to-speech unavailable (check SARVAM_API_KEY and credits)",
        )
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="eka.wav"',
            "Cache-Control": "no-store",
            "X-Eka-Voice": payload.mode,
        },
    )


@router.get("/voices")
async def list_voices():
    """Which Sarvam speaker each mode maps to."""
    health = await tts_service.health()
    return {"voices": health["voices"], "configured": health["configured"]}
