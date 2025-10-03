import torch
import whisper


class WhisperModelLoader:
    _instance: "WhisperModelLoader | None" = None

    def __new__(cls) -> "WhisperModelLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def get_model(self) -> whisper.Whisper:
        if self._model is None:
            self._model = whisper.load_model("base")
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
        return self._model


def get_whisper_model():
    return WhisperModelLoader().get_model()
