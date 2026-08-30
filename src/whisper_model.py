import logging
import os
import threading
import time

import numpy as np
from faster_whisper import WhisperModel


logger = logging.getLogger(__name__)


class WhisperModelLoader:
    _instance: "WhisperModelLoader | None" = None

    def __new__(cls) -> "WhisperModelLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._lock = threading.Lock()
            cls._instance._loaded_at = None
        return cls._instance

    def get_model(self) -> WhisperModel:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            model_name = os.getenv("WHISPER_MODEL", "base")
            device = os.getenv("WHISPER_DEVICE", "cpu")
            compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
            cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "0"))
            num_workers = int(os.getenv("WHISPER_NUM_WORKERS", "1"))

            start_time = time.time()
            logger.info(
                "Loading faster-whisper model '%s' on %s with compute_type=%s",
                model_name,
                device,
                compute_type,
            )
            self._model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            )
            self._loaded_at = time.time()
            logger.info(
                "Loaded faster-whisper model '%s' in %.2fs",
                model_name,
                self._loaded_at - start_time,
            )
            return self._model

    def warmup(self, language: str | None = None) -> dict:
        start_time = time.time()
        model = self.get_model()
        warmup_language = (language or "en").strip().lower() or "en"
        if warmup_language in {"nb", "nn", "nb-no", "nn-no", "nor"}:
            warmup_language = "no"
        if warmup_language in {"zh-cn", "zh-hans", "zh_cn", "cmn", "mandarin", "chinese"}:
            warmup_language = "zh"
        warmup_language = warmup_language.split("-")[0]
        # Force lazy runtime initialization inside CTranslate2 by running a tiny decode.
        segments, info = model.transcribe(
            np.zeros(16_000, dtype=np.float32),
            beam_size=1,
            language=warmup_language,
            vad_filter=False,
        )
        list(segments)
        return {
            "success": True,
            "model": os.getenv("WHISPER_MODEL", "base"),
            "device": os.getenv("WHISPER_DEVICE", "cpu"),
            "compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            "language": info.language,
            "requested_language": warmup_language,
            "warmup_time_seconds": round(time.time() - start_time, 3),
            "loaded": self._model is not None,
        }


def get_whisper_model() -> WhisperModel:
    return WhisperModelLoader().get_model()


def warmup_whisper_model(language: str | None = None) -> dict:
    return WhisperModelLoader().warmup(language)
