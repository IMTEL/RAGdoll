# transcribe_service.py
from flask import Flask, request, jsonify

import whisper
import numpy as np
# import ffmpeg
from fastapi import UploadFile, HTTPException
import tempfile
import logging

import os

from src.config import Config

model = Config().whisper_model



app = Flask(__name__)


import io, math, numpy as np, soundfile as sf
from scipy.signal import resample_poly      # pip install scipy soundfile

TARGET_SR = 16_000          # 16 kHz mono float32

def load_audio_from_upload(file) -> np.ndarray:
    try:
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
    except sf.SoundFileError as e:
        logging.error(f"SoundFile error when processing audio: {str(e)}")
        raise ValueError(f"Invalid audio file format: {str(e)}")
    except Exception as e:
        logging.error(f"Error loading audio: {str(e)}")
        raise ValueError(f"Failed to process audio file: {str(e)}")
        
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
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text

def transcribe_audio(file: UploadFile, language: str = None) -> dict:
    """
    Transcribe an audio file with specified language.
    
    Args:
        file (UploadFile): The audio file to transcribe
        language (str, optional): Language code (e.g., 'en', 'es', 'fr')
        
    Returns:
        dict: Response containing transcription or error message
    """
    try:
        import time
        start_time = time.time()
        
        # Check file size (limit to 25MB for example)
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > 25 * 1024 * 1024:  # 25MB
            return {
                "success": False,
                "error": "File too large. Maximum size is 25MB."
            }
            
        # Load audio
        audio = load_audio_from_upload(file)
        audio = whisper.pad_or_trim(audio)
        
        # Process with whisper
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Set language in options if provided
        if language:
            options = whisper.DecodingOptions(language=language)
        else:
            options = whisper.DecodingOptions()
            
        result = whisper.decode(model, mel, options)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add server identifier to the response
        return {
            "success": True,
            "transcription": result.text,
            "server_processed": True,
            "processing_time_seconds": round(processing_time, 3),
            "processor": "Server-based Whisper"
        }
        
    except ValueError as e:
        # Handle format errors
        return {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        # Handle other errors
        logging.error(f"Transcription error: {str(e)}")
        return {
            "success": False, 
            "error": f"Failed to transcribe audio: {str(e)}"
        }

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