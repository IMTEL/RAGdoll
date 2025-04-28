# transcribe_service.py
from flask import Flask, request, jsonify

from faster_whisper import WhisperModel
import numpy as np
# import ffmpeg
from fastapi import UploadFile
import tempfile
import logging
import os
import torch

from src.config import Config

# Configure logging for faster-whisper
logging.basicConfig()
whisper_logger = logging.getLogger("faster_whisper")
whisper_logger.setLevel(logging.INFO)

# Check for GPU
device = "cuda" if torch.cuda.is_available() else Config().whisper_device
compute_type = "float16" if torch.cuda.is_available() else Config().whisper_compute_type

# Load the faster-whisper model with advanced optimizations
model = WhisperModel(
    Config().whisper_model_size,
    device=device,
    compute_type=compute_type,
    # Adding optimizations:
    cpu_threads=8,            # Increased for better parallelization
    num_workers=4,            # Increased for parallel processing
    download_root=os.path.join(os.path.dirname(__file__), "models", "whisper")  # Cache models locally
)

# Configure model to be faster by reducing beam size and using other optimizations
DEFAULT_BEAM_SIZE = 1        # Reduced from 5 for faster processing
DEFAULT_BEST_OF = 1          # Reduced for faster processing
USE_EFFICIENT_BY_DEFAULT = True # Use efficient processing by default

app = Flask(__name__)


import io, math, numpy as np, soundfile as sf
from scipy.signal import resample_poly      # pip install scipy soundfile

TARGET_SR = 16_000          # 16 kHz mono float32

def load_audio_from_upload(file) -> np.ndarray:
    raw = file.file.read()                  # UploadFile → bytes
    # --- decode ----------------------------------------------------------------------------------
    with io.BytesIO(raw) as bio:
        audio, sr = sf.read(bio, dtype='float32')   # libsndfile does the heavy lifting
    # --- mono ------------------------------------------------------------------------------------
    if audio.ndim > 1:
        audio = audio.mean(axis=1)          # down-mix
    # --- resample -------------------------------------------------------------------------------
    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)         # polyphase → good quality & fast
        audio = resample_poly(audio, TARGET_SR // g, sr // g).astype('float32')
    return audio


# def load_audio_from_upload(file: UploadFile) -> np.ndarray:
#     input_bytes = file.file.read()

#     # Convert uploaded file (WAV/MP3/etc) into 16kHz float32 mono PCM
#     out, _ = (
#         ffmpeg
#         .input("pipe:0")
#         .output("pipe:1", format="f32le", ac=1, ar="16000")
#         .run(input=input_bytes, capture_stdout=True, capture_stderr=True)
#     )
#     audio = np.frombuffer(out, np.float32)
#     return audio


def transcribe_from_upload(file: UploadFile) -> str:
    audio = load_audio_from_upload(file)
    # Use the batched model for faster processing
    segments, info = model.transcribe(
        audio, 
        beam_size=5,
        vad_filter=True,  # Filter out non-speech parts
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    
    # Concatenate segments to get the full text
    full_text = "".join(segment.text for segment in segments)
    return full_text


def transcribe(audio_path: str) -> str:
    """
    Transcribe an audio file using the optimized faster-whisper model.
    
    Args:
        audio_path (str): Path to the audio file.
        
    Returns:
        str: Transcribed text.
    """
    segments, info = model.transcribe(
        audio_path, 
        beam_size=5, 
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    full_text = "".join(segment.text for segment in segments)
    return full_text