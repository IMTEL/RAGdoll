"""Local text-to-speech service abstraction."""

from __future__ import annotations

import base64
import re
import time
import wave
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import BoundedSemaphore, Lock

from src.config import Config


# Piper is CPU heavy: at most TTS_MAX_CONCURRENCY syntheses run at once, the
# rest wait. Shared by every synthesis in the process.
TTS_SEMAPHORE = BoundedSemaphore(Config().TTS_MAX_CONCURRENCY)


@dataclass(frozen=True)
class SpeechResult:
    audio_base64: str
    format: str
    mime_type: str
    engine: str
    voice: str
    language: str
    processing_time_seconds: float

    def to_dict(self) -> dict:
        return {
            "audio_base64": self.audio_base64,
            "format": self.format,
            "mime_type": self.mime_type,
            "engine": self.engine,
            "voice": self.voice,
            "language": self.language,
            "processing_time_seconds": self.processing_time_seconds,
        }


class TextToSpeechService(ABC):
    @abstractmethod
    def synthesize(self, text: str, language: str | None = None) -> SpeechResult:
        """Synthesize text into speech audio."""

    @abstractmethod
    def warmup(self, language: str | None = None) -> dict:
        """Load and warm the speech model for a language."""


class PiperTextToSpeechService(TextToSpeechService):
    """Piper implementation backed by local ONNX voice files."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.voice_dir = Path(self.config.TTS_VOICE_DIR)
        self.use_cuda = self.config.TTS_USE_CUDA
        self._voices: dict[str, object] = {}
        self._lock = Lock()

    def synthesize(self, text: str, language: str | None = None) -> SpeechResult:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Text is required for speech synthesis.")

        normalized_language = self._normalize_language(language)
        voice_name = self._voice_name_for_language(normalized_language)
        voice = self._load_voice(voice_name)

        wav_buffer = BytesIO()
        with TTS_SEMAPHORE:
            start_time = time.time()
            with wave.open(wav_buffer, "wb") as wav_file:
                voice.synthesize_wav(clean_text, wav_file)

        return SpeechResult(
            audio_base64=base64.b64encode(wav_buffer.getvalue()).decode("ascii"),
            format="wav",
            mime_type="audio/wav",
            engine="piper",
            voice=voice_name,
            language=normalized_language,
            processing_time_seconds=round(time.time() - start_time, 3),
        )

    def warmup(self, language: str | None = None) -> dict:
        normalized_language = self._normalize_language(language)
        start_time = time.time()
        result = self.synthesize(self.config.TTS_WARMUP_TEXT, normalized_language)
        return {
            "success": True,
            "engine": result.engine,
            "voice": result.voice,
            "language": result.language,
            "format": result.format,
            "warmup_time_seconds": round(time.time() - start_time, 3),
        }

    def _normalize_language(self, language: str | None) -> str:
        value = (language or self.config.TTS_DEFAULT_LANGUAGE or "en").strip().lower()
        if value in {"nb", "nn", "nor", "no_no", "nb_no", "nn_no"}:
            return "no"
        if value in {"en_us", "en-gb", "eng"}:
            return "en"
        if value in {"es_es", "spa"}:
            return "es"
        if value in {"zh", "zh-cn", "zh_cn", "cmn", "mandarin", "chinese"}:
            return "zh"
        return value.split("-")[0].split("_")[0] or "en"

    def _voice_name_for_language(self, language: str) -> str:
        voice_map = {
            "en": self.config.TTS_DEFAULT_VOICE_EN,
            "no": self.config.TTS_DEFAULT_VOICE_NO,
            "es": self.config.TTS_DEFAULT_VOICE_ES,
            "zh": self.config.TTS_DEFAULT_VOICE_ZH,
        }
        return voice_map.get(
            language, voice_map.get(self.config.TTS_DEFAULT_LANGUAGE, "")
        )

    def _load_voice(self, voice_name: str):
        if not voice_name:
            raise ValueError("No Piper voice is configured for this language.")

        with self._lock:
            if voice_name in self._voices:
                return self._voices[voice_name]

            try:
                from piper import PiperVoice
            except ImportError as exc:
                raise RuntimeError(
                    "Piper TTS is not installed. Install the piper-tts package."
                ) from exc

            model_path = self._resolve_voice_model_path(voice_name)
            voice = PiperVoice.load(str(model_path), use_cuda=self.use_cuda)
            self._voices[voice_name] = voice
            return voice

    def _resolve_voice_model_path(self, voice_name: str) -> Path:
        candidate = Path(voice_name)
        if candidate.suffix == ".onnx" and candidate.exists():
            return candidate

        direct_path = self.voice_dir / (
            voice_name if voice_name.endswith(".onnx") else f"{voice_name}.onnx"
        )
        if direct_path.exists():
            return direct_path

        if self.voice_dir.exists():
            matches = list(self.voice_dir.rglob(direct_path.name))
            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"Piper voice '{voice_name}' was not found in '{self.voice_dir}'. "
            "Install the matching .onnx and .onnx.json voice files there."
        )


class UnsupportedTextToSpeechService(TextToSpeechService):
    def __init__(self, engine: str):
        self.engine = engine

    def synthesize(self, text: str, language: str | None = None) -> SpeechResult:
        raise RuntimeError(f"Unsupported TTS engine '{self.engine}'.")

    def warmup(self, language: str | None = None) -> dict:
        raise RuntimeError(f"Unsupported TTS engine '{self.engine}'.")


_service: TextToSpeechService | None = None
_service_lock = Lock()


def get_tts_service() -> TextToSpeechService:
    global _service
    with _service_lock:
        if _service is not None:
            return _service

        config = Config()
        if config.TTS_ENGINE == "piper":
            _service = PiperTextToSpeechService(config)
        else:
            _service = UnsupportedTextToSpeechService(config.TTS_ENGINE)

        return _service


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=\S)")
_ABBREVIATIONS = (
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "st.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on ``.!?`` followed by whitespace (best effort).

    Pieces ending in a known abbreviation are glued to the next sentence.
    """
    clean = (text or "").strip()
    if not clean:
        return []
    sentences: list[str] = []
    carry = ""
    for piece in _SENTENCE_BOUNDARY.split(clean):
        piece = f"{carry} {piece}".strip() if carry else piece.strip()
        carry = ""
        if piece.split()[-1].lower() in _ABBREVIATIONS:
            carry = piece
            continue
        if piece:
            sentences.append(piece)
    if carry:
        sentences.append(carry)
    return sentences


def _read_wav(audio_base64: str) -> tuple[tuple, bytes]:
    with wave.open(BytesIO(base64.b64decode(audio_base64)), "rb") as wav_file:
        params = (
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getframerate(),
        )
        return params, wav_file.readframes(wav_file.getnframes())


def _sample_rate(audio_base64: str) -> int | None:
    try:
        return _read_wav(audio_base64)[0][2]
    except (wave.Error, ValueError, EOFError):
        return None


def concat_wav_base64(audio_parts: list[str]) -> str:
    """Join base64 WAVs into one WAV: a single header over concatenated PCM."""
    params = None
    frames = []
    for part in audio_parts:
        part_params, part_frames = _read_wav(part)
        if params is None:
            params = part_params
        elif part_params != params:
            raise ValueError("Cannot concatenate WAV segments with different formats.")
        frames.append(part_frames)
    channels, sample_width, rate = params
    out = BytesIO()
    with wave.open(out, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"".join(frames))
    return base64.b64encode(out.getvalue()).decode("ascii")


def synthesize_sentences(
    service: TextToSpeechService, text: str, language: str | None = None
) -> tuple[dict, list[dict]]:
    """Synthesize each sentence in parallel; return (speech, speech_segments).

    Concurrency is bounded by TTS_SEMAPHORE (acquired inside each synthesize
    call, never held while waiting on other sentences). ``speech`` is the
    concatenation of all segments; for a one-sentence answer it is exactly the
    single synthesis result.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        result = service.synthesize(sentences[0] if sentences else text, language)
        speech = result.to_dict()
        segment = {
            "text": sentences[0] if sentences else text.strip(),
            "audio_base64": speech.get("audio_base64"),
            "sample_rate": _sample_rate(speech.get("audio_base64") or ""),
        }
        return speech, [segment]

    workers = min(len(sentences), max(1, Config().TTS_MAX_CONCURRENCY))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda s: service.synthesize(s, language), sentences))

    dicts = [r.to_dict() for r in results]
    segments = [
        {
            "text": sentence,
            "audio_base64": d["audio_base64"],
            "sample_rate": _sample_rate(d["audio_base64"]),
        }
        for sentence, d in zip(sentences, dicts, strict=True)
    ]
    speech = dict(dicts[0])
    speech["audio_base64"] = concat_wav_base64([d["audio_base64"] for d in dicts])
    speech["processing_time_seconds"] = round(
        sum(d.get("processing_time_seconds", 0) for d in dicts), 3
    )
    return speech, segments
