import io
import logging
import math
import os
import tempfile
import time

import numpy as np
import soundfile as sf
from fastapi import UploadFile
from flask import Flask
from scipy.signal import resample_poly

from src.whisper_model import get_whisper_model


logger = logging.getLogger(__name__)


# TODO: switch to FastAPI
app = Flask(__name__)

TARGET_SR = 16_000


def _save_audio_to_temp(raw: bytes, filename: str | None = None) -> str:
    suffix = os.path.splitext(filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(raw)
        return temp_file.name


def _decode_audio_with_soundfile(raw: bytes) -> np.ndarray:
    with io.BytesIO(raw) as bio:
        audio, sr = sf.read(bio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g).astype("float32")
    return audio


def load_audio_from_upload(file) -> np.ndarray:
    raw = file.file.read()
    try:
        return _decode_audio_with_soundfile(raw)
    except sf.SoundFileError as e:
        logger.info(
            "SoundFile could not decode uploaded audio: %s",
            e,
        )
        raise ValueError(
            "Invalid audio file format for direct decoding. Use transcribe_audio for browser recordings."
        ) from e
    except Exception as e:
        logger.error(f"Error loading audio: {e!s}")
        raise ValueError("Failed to process audio file.") from e


def _transcribe_input(audio_input: np.ndarray | str, language: str | None = None) -> dict:
    model = get_whisper_model()
    segments, info = model.transcribe(
        audio_input,
        language=language or None,
        beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "1")),
        vad_filter=os.getenv("WHISPER_VAD_FILTER", "false").lower() == "true",
    )
    text = "".join(segment.text for segment in segments).strip()
    return {
        "text": text,
        "language": info.language,
        "language_probability": round(float(info.language_probability or 0), 4),
    }


def _transcribe_upload(file: UploadFile, language: str | None = None) -> dict:
    raw = file.file.read()
    try:
        audio_input: np.ndarray | str = _decode_audio_with_soundfile(raw)
        temp_path = None
    except sf.SoundFileError:
        temp_path = _save_audio_to_temp(raw, getattr(file, "filename", None))
        audio_input = temp_path
    except Exception as e:
        logger.error(f"Error loading audio: {e!s}")
        raise ValueError("Failed to process audio file.") from e

    try:
        return _transcribe_input(audio_input, language)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                logger.warning(f"Failed to remove temporary audio file: {temp_path}")


def transcribe_from_upload(file: UploadFile) -> str:
    return _transcribe_upload(file)["text"]


def transcribe_audio(file: UploadFile, language: str | None = None) -> dict:
    """Transcribe an audio file with specified language.

    Args:
        file: The audio file to transcribe.
        language: Optional language code, for example "en", "es", "fr", or "no".

    Returns:
        Response containing transcription or an error message.
    """
    try:
        start_time = time.time()

        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size > 25 * 1024 * 1024:
            return {"success": False, "error": "File too large. Maximum size is 25MB."}

        result = _transcribe_upload(file, language)
        processing_time = time.time() - start_time

        return {
            "success": True,
            "transcription": result["text"],
            "language": result["language"],
            "language_probability": result["language_probability"],
            "server_processed": True,
            "processing_time_seconds": round(processing_time, 3),
            "processor": "Server-based faster-whisper",
        }

    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Transcription error: {e!s}")
        return {"success": False, "error": f"Failed to transcribe audio: {e!s}"}


def transcribe(audio):
    """Transcribe an audio file using the configured faster-whisper model."""
    return _transcribe_input(audio)["text"]
