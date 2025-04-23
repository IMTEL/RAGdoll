# transcribe_service.py
from flask import Flask, request, jsonify

import whisper
import numpy as np
import ffmpeg
from fastapi import UploadFile
import tempfile

import os

from src.config import Config

model = Config().whisper_model



app = Flask(__name__)

def load_audio_from_upload(file: UploadFile) -> np.ndarray:
    input_bytes = file.file.read()

    # Convert uploaded file (WAV/MP3/etc) into 16kHz float32 mono PCM
    out, _ = (
        ffmpeg
        .input("pipe:0")
        .output("pipe:1", format="f32le", ac=1, ar="16000")
        .run(input=input_bytes, capture_stdout=True, capture_stderr=True)
    )
    audio = np.frombuffer(out, np.float32)
    return audio


def transcribe_from_upload(file: UploadFile) -> str:
    audio = load_audio_from_upload(file)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text


def transcribe(audio):
    """
    Transcribe an audio file using Whisper model.
    
    Args:
        audio (str): Path to the audio file.
        
    Returns:
        str: Transcribed text.
    """
    # model = Config.whisper_model

    result = model.transcribe(audio)
    return result["text"]