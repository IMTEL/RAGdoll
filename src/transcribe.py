import io
import logging
import math
import os
import tempfile

import numpy as np
import soundfile as sf
import whisper
from fastapi import UploadFile
from flask import Flask
from scipy.signal import resample_poly

from src.whisper_model import get_whisper_model


logger = logging.getLogger(__name__)


model = get_whisper_model()

# TODO: switch to FastAPI
app = Flask(__name__)

TARGET_SR = 16_000


def _load_audio_with_whisper(raw: bytes, filename: str | None = None) -> np.ndarray:
    suffix = os.path.splitext(filename or "")[1] or ".webm"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(raw)
            temp_path = temp_file.name
        return whisper.load_audio(temp_path, sr=TARGET_SR).astype("float32")
    except Exception as e:
        logger.error(f"Whisper/ffmpeg fallback failed when processing audio: {e!s}")
        raise ValueError(
            "Invalid audio file format. Browser recordings require ffmpeg support in the backend container."
        ) from e
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                logger.warning(f"Failed to remove temporary audio file: {temp_path}")


def load_audio_from_upload(file) -> np.ndarray:
    raw = file.file.read()
    try:
        with io.BytesIO(raw) as bio:
            audio, sr = sf.read(bio, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            g = math.gcd(sr, TARGET_SR)
            audio = resample_poly(audio, TARGET_SR // g, sr // g).astype("float32")
        return audio
    except sf.SoundFileError as e:
        logger.info(
            "SoundFile could not decode uploaded audio, trying Whisper/ffmpeg fallback: %s",
            e,
        )
        return _load_audio_with_whisper(raw, getattr(file, "filename", None))
    except Exception as e:
        logger.error(f"Error loading audio: {e!s}")
        raise ValueError("Failed to process audio file.") from e


def transcribe_from_upload(file: UploadFile) -> str:
    audio = load_audio_from_upload(file)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text


def transcribe_audio(file: UploadFile, language: str | None = None) -> dict:
    """Transcribe an audio file with specified language.

    Args:
        file: The audio file to transcribe.
        language: Optional language code, for example "en", "es", or "fr".

    Returns:
        Response containing transcription or an error message.
    """
    try:
        import time

        start_time = time.time()

        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size > 25 * 1024 * 1024:
            return {"success": False, "error": "File too large. Maximum size is 25MB."}

        audio = load_audio_from_upload(file)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = (
            whisper.DecodingOptions(language=language)
            if language
            else whisper.DecodingOptions()
        )
        result = whisper.decode(model, mel, options)

        processing_time = time.time() - start_time

        return {
            "success": True,
            "transcription": result.text,
            "server_processed": True,
            "processing_time_seconds": round(processing_time, 3),
            "processor": "Server-based Whisper",
        }

    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Transcription error: {e!s}")
        return {"success": False, "error": f"Failed to transcribe audio: {e!s}"}


def transcribe(audio):
    """Transcribe an audio file using Whisper model.

    Args:
        audio: Path to the audio file.

    Returns:
        Transcribed text.
    """
    result = model.transcribe(audio)
    return result["text"]
