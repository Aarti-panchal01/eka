"""Text-to-speech via Sarvam Bulbul — one voice per Eka persona.

Returns WAV bytes, or None. None is a normal outcome, not an error: the caller
returns the text reply anyway and the UI simply doesn't speak. Voice is an
enhancement, never a dependency.

Long replies are split at sentence boundaries and the resulting WAV chunks are
merged at the PCM level (naively concatenating WAV files yields a stream with
headers buried mid-body, which most players truncate at the first chunk).
"""

import asyncio
import base64
import io
import logging
import re
import wave
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import httpx

from config import settings

logger = logging.getLogger(__name__)

# Split after . ! ? … or a Devanagari danda, when followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?।…])\s+")


class TTSService:
    def __init__(self) -> None:
        self._cache: "OrderedDict[Tuple[str, str], bytes]" = OrderedDict()
        self._last_error: Optional[str] = None
        self._available: Optional[bool] = None

    # ---------------------------------------------------------- public API
    async def synthesize(self, text: str, mode: str = "reflection") -> Optional[bytes]:
        """Speak `text` in `mode`'s voice. Returns WAV bytes or None."""
        text = self._clean(text)
        if not text:
            return None

        if not settings.SARVAM_API_KEY:
            logger.debug("SARVAM_API_KEY not set — skipping TTS")
            return None

        cache_key = (text[:100], mode)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        speaker = settings.voice_for_mode(mode)
        chunks = self._chunk(text, settings.TTS_CHUNK_CHARS)

        audio_parts: List[bytes] = []
        for index, chunk in enumerate(chunks):
            part = await self._request(chunk, speaker)
            if part is None:
                # Partial audio beats no audio — return what we have.
                if audio_parts:
                    logger.warning(
                        "TTS failed on chunk %d/%d — returning partial audio",
                        index + 1, len(chunks),
                    )
                    break
                self._available = False
                return None
            audio_parts.append(part)

        if not audio_parts:
            return None

        audio = audio_parts[0] if len(audio_parts) == 1 else self._merge_wav(audio_parts)
        self._available = True

        self._cache[cache_key] = audio
        if len(self._cache) > settings.TTS_CACHE_SIZE:
            self._cache.popitem(last=False)
        return audio

    # --------------------------------------------------------------- HTTP
    async def _request(self, text: str, speaker: str) -> Optional[bytes]:
        headers = {
            "api-subscription-key": settings.SARVAM_API_KEY,
            "Content-Type": "application/json",
        }

        # Sarvam has shipped two request shapes across model versions. Try the
        # documented v4 shape first, then the newer single-`text` shape.
        payloads = [
            {
                "inputs": [text],
                "target_language_code": settings.SARVAM_LANGUAGE,
                "speaker": speaker,
                "model": settings.SARVAM_TTS_MODEL,
                "speech_sample_rate": settings.SARVAM_SAMPLE_RATE,
                "enable_preprocessing": True,
            },
            {
                "text": text,
                "target_language_code": settings.SARVAM_LANGUAGE,
                "speaker": speaker,
                "model": settings.SARVAM_TTS_MODEL,
                "speech_sample_rate": settings.SARVAM_SAMPLE_RATE,
                "enable_preprocessing": True,
            },
        ]

        for attempt, payload in enumerate(payloads):
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
                    response = await client.post(
                        settings.SARVAM_TTS_URL, headers=headers, json=payload
                    )

                if response.status_code == 200:
                    return self._decode(response.json())

                if response.status_code in (400, 422) and attempt == 0:
                    logger.info("Sarvam rejected the v4 payload shape — trying `text`")
                    continue

                if response.status_code == 429:
                    await asyncio.sleep(2)
                    continue

                self._last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.warning("Sarvam TTS %s", self._last_error)
                return None

            except httpx.TimeoutException:
                self._last_error = "timeout"
                logger.warning("Sarvam TTS timed out")
                return None
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("Sarvam TTS error: %s", exc)
                return None
        return None

    @staticmethod
    def _decode(body: Dict) -> Optional[bytes]:
        """Sarvam returns base64 audio under `audios` (or `audio`)."""
        payload = body.get("audios") or body.get("audio")
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if not payload or not isinstance(payload, str):
            logger.warning("Unexpected Sarvam TTS response keys: %s", list(body.keys()))
            return None
        try:
            return base64.b64decode(payload)
        except Exception as exc:
            logger.warning("Sarvam base64 decode failed: %s", exc)
            return None

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _clean(text: str) -> str:
        """Strip markdown so the voice doesn't read asterisks aloud."""
        if not text:
            return ""
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
        text = re.sub(r"[*_#>]", " ", text)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _chunk(text: str, limit: int) -> List[str]:
        """Split at sentence boundaries, packing up to `limit` chars per chunk."""
        if len(text) <= limit:
            return [text]

        chunks: List[str] = []
        current = ""
        for sentence in _SENTENCE_END.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            # A single sentence longer than the limit gets hard-split on words.
            if len(sentence) > limit:
                if current:
                    chunks.append(current)
                    current = ""
                words, buffer = sentence.split(), ""
                for word in words:
                    if len(buffer) + len(word) + 1 > limit:
                        chunks.append(buffer)
                        buffer = word
                    else:
                        buffer = f"{buffer} {word}".strip()
                if buffer:
                    current = buffer
                continue

            if len(current) + len(sentence) + 1 > limit:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()

        if current:
            chunks.append(current)
        return chunks or [text[:limit]]

    @staticmethod
    def _merge_wav(parts: List[bytes]) -> bytes:
        """Concatenate WAV chunks properly: one header, all frames."""
        try:
            frames: List[bytes] = []
            params = None
            for part in parts:
                with wave.open(io.BytesIO(part), "rb") as reader:
                    if params is None:
                        params = reader.getparams()
                    frames.append(reader.readframes(reader.getnframes()))

            if params is None:
                return b"".join(parts)

            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as writer:
                writer.setnchannels(params.nchannels)
                writer.setsampwidth(params.sampwidth)
                writer.setframerate(params.framerate)
                writer.writeframes(b"".join(frames))
            return buffer.getvalue()
        except Exception as exc:
            # Not WAV (Sarvam may return mp3/opus). Raw concat is the best we
            # can do, and single-chunk replies — the common case — are unaffected.
            logger.debug("WAV merge failed (%s) — falling back to raw concat", exc)
            return b"".join(parts)

    # ------------------------------------------------------------- health
    async def health(self) -> Dict:
        return {
            "configured": bool(settings.SARVAM_API_KEY),
            "available": self._available,
            "cache_size": len(self._cache),
            "last_error": self._last_error,
            "voices": {
                mode: settings.voice_for_mode(mode)
                for mode in ("founder", "chanakya", "gita", "reflection")
            },
        }

    async def probe(self) -> bool:
        """Live check with a two-word utterance. Used by scripts/check_build_status."""
        if not settings.SARVAM_API_KEY:
            return False
        audio = await self.synthesize("Hello.", "founder")
        return bool(audio and len(audio) > 1000)


tts_service = TTSService()
