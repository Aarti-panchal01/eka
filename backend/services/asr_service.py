"""Speech-to-text via Sarvam Saarika, falling back to local faster-whisper.

    Sarvam (en-IN)  ->  Sarvam (auto-detect, if confidence was low)
                    ->  faster-whisper base on CPU, if installed
                    ->  HTTPException 503 with an actionable message

Unlike TTS, ASR cannot degrade silently: if we can't transcribe, the user's
words are simply gone, so the last tier raises rather than returning "".
"""

import asyncio
import logging
from typing import Dict, Optional, Tuple

import httpx

from config import settings

logger = logging.getLogger(__name__)

LOW_CONFIDENCE = 0.6
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25MB


class ASRService:
    def __init__(self) -> None:
        self._whisper: Optional[object] = None
        self._whisper_tried = False
        self._lock = asyncio.Lock()
        self._last_error: Optional[str] = None
        self._last_backend: Optional[str] = None

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
    ) -> str:
        """Audio bytes -> transcript. Raises HTTPException if all tiers fail."""
        from fastapi import HTTPException

        if not audio_bytes:
            return ""
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio too large ({len(audio_bytes) / 1e6:.1f}MB, max 25MB)",
            )

        if settings.SARVAM_API_KEY:
            # Caller language wins; SARVAM_LANGUAGE is only the fallback.
            declared = language or settings.SARVAM_LANGUAGE
            transcript, confidence = await self._sarvam(
                audio_bytes, filename, declared
            )
            if transcript is not None:
                # A low-confidence result usually means the speaker wasn't
                # using the language we declared. Retry with auto-detect.
                if confidence is not None and confidence < LOW_CONFIDENCE:
                    logger.info(
                        "Sarvam confidence %.2f — retrying with auto-detect", confidence
                    )
                    retry, retry_confidence = await self._sarvam(
                        audio_bytes, filename, "unknown"
                    )
                    if retry and (retry_confidence or 0) >= (confidence or 0):
                        self._last_backend = "sarvam:auto"
                        return retry.strip()
                self._last_backend = "sarvam"
                return transcript.strip()
        else:
            logger.debug("SARVAM_API_KEY not set — going straight to whisper")

        whisper_text = await self._whisper_transcribe(audio_bytes)
        if whisper_text is not None:
            self._last_backend = "faster-whisper"
            return whisper_text.strip()

        raise HTTPException(
            status_code=503,
            detail=(
                "Speech-to-text unavailable. Either set SARVAM_API_KEY, or "
                "install a local fallback with `pip install faster-whisper`. "
                f"Last error: {self._last_error}"
            ),
        )

    # ------------------------------------------------------------- sarvam
    async def _sarvam(
        self, audio_bytes: bytes, filename: str, language: str
    ) -> Tuple[Optional[str], Optional[float]]:
        headers = {"api-subscription-key": settings.SARVAM_API_KEY}
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {
            "model": settings.SARVAM_STT_MODEL,
            "language_code": language,
            "with_timestamps": "false",
        }

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    settings.SARVAM_STT_URL, headers=headers, files=files, data=data
                )

            if response.status_code == 200:
                body = response.json()
                transcript = (
                    body.get("transcript")
                    or body.get("text")
                    or body.get("transcription")
                    or ""
                )
                confidence = body.get("confidence")
                try:
                    confidence = float(confidence) if confidence is not None else None
                except (TypeError, ValueError):
                    confidence = None
                return transcript, confidence

            self._last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            logger.warning("Sarvam STT %s", self._last_error)
            return None, None

        except httpx.TimeoutException:
            self._last_error = "Sarvam STT timeout"
            logger.warning(self._last_error)
            return None, None
        except Exception as exc:
            self._last_error = f"Sarvam STT error: {exc}"
            logger.warning(self._last_error)
            return None, None

    # ------------------------------------------------------------ whisper
    async def _whisper_transcribe(self, audio_bytes: bytes) -> Optional[str]:
        model = await self._get_whisper()
        if model is None:
            return None

        import tempfile
        from pathlib import Path

        temp_path = None
        try:
            # faster-whisper needs a real file path (it decodes via ffmpeg).
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                handle.write(audio_bytes)
                temp_path = handle.name

            def run() -> str:
                segments, _info = model.transcribe(temp_path, beam_size=5)
                return " ".join(segment.text for segment in segments)

            return await asyncio.to_thread(run)
        except Exception as exc:
            self._last_error = f"whisper error: {exc}"
            logger.warning(self._last_error)
            return None
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    async def _get_whisper(self):
        if self._whisper_tried:
            return self._whisper

        async with self._lock:
            if self._whisper_tried:
                return self._whisper
            self._whisper_tried = True
            try:
                from faster_whisper import WhisperModel

                logger.info("Loading faster-whisper base (int8, CPU)...")
                self._whisper = await asyncio.to_thread(
                    WhisperModel, "base", device="cpu", compute_type="int8"
                )
                logger.info("faster-whisper ready")
            except ImportError:
                logger.info(
                    "faster-whisper not installed — no local ASR fallback. "
                    "This is expected on Render free tier (the model is ~150MB)."
                )
                self._whisper = None
            except Exception as exc:
                logger.warning("faster-whisper failed to load: %s", exc)
                self._whisper = None
        return self._whisper

    # ------------------------------------------------------------- health
    async def health(self) -> Dict:
        return {
            "sarvam_configured": bool(settings.SARVAM_API_KEY),
            "whisper_loaded": self._whisper is not None,
            "last_backend": self._last_backend,
            "last_error": self._last_error,
        }


asr_service = ASRService()
