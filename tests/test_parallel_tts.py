"""Sentence-parallel TTS: ordering, segments, valid concatenated WAV."""

import base64
import threading
import time
import wave
from io import BytesIO

import pytest

from src import tts as tts_module
from src.tts import SpeechResult, split_sentences, synthesize_sentences


def _wav_b64(n_frames: int, rate: int = 22050) -> str:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * n_frames)
    return base64.b64encode(buf.getvalue()).decode()


class FakeService:
    """Stub synthesizer: frame count = len(text); earlier sentences are slower."""

    def __init__(self):
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.calls = []

    def synthesize(self, text, language=None):
        self.calls.append(text)
        with tts_module.TTS_SEMAPHORE:
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.05 if text.startswith("One") else 0.0)
            with self.lock:
                self.active -= 1
        return SpeechResult(
            _wav_b64(len(text)), "wav", "audio/wav", "piper", "v", "en", 0.1
        )


def test_split_sentences_basic_and_abbreviation():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("Ask Dr. Smith now. Then go.") == [
        "Ask Dr. Smith now.",
        "Then go.",
    ]
    assert split_sentences("  ") == []


def test_order_preserved_and_segments_match():
    svc = FakeService()
    text = "One is slow. Two is fast. Three too."
    _speech, segments = synthesize_sentences(svc, text, "en")
    assert [s["text"] for s in segments] == [
        "One is slow.",
        "Two is fast.",
        "Three too.",
    ]
    for seg in segments:
        assert seg["sample_rate"] == 22050
        with wave.open(BytesIO(base64.b64decode(seg["audio_base64"]))) as w:
            assert w.getnframes() == len(seg["text"])


def test_concatenated_speech_is_single_valid_wav():
    svc = FakeService()
    sentences = ["One a.", "Two bb.", "Three ccc."]
    speech, _ = synthesize_sentences(svc, " ".join(sentences))
    raw = base64.b64decode(speech["audio_base64"])
    assert raw.count(b"RIFF") == 1
    with wave.open(BytesIO(raw)) as w:
        assert w.getnframes() == sum(len(s) for s in sentences)
        assert w.getframerate() == 22050
    assert speech["format"] == "wav"


def test_one_sentence_matches_previous_behaviour():
    svc = FakeService()
    speech, segments = synthesize_sentences(svc, "Just one sentence.")
    expected = SpeechResult(
        _wav_b64(len("Just one sentence.")), "wav", "audio/wav", "piper", "v", "en", 0.1
    ).to_dict()
    assert speech == expected
    assert len(segments) == 1
    assert segments[0]["audio_base64"] == expected["audio_base64"]


def test_parallel_but_bounded_by_semaphore(monkeypatch):
    monkeypatch.setattr(tts_module, "TTS_SEMAPHORE", threading.BoundedSemaphore(2))
    svc = FakeService()
    synthesize_sentences(svc, "One a. Two b. Three c. Four d. Five e.")
    assert 1 < svc.peak <= 2


def test_empty_text_still_raises_like_before():
    class Raising:
        def synthesize(self, text, language=None):
            raise ValueError("Text is required for speech synthesis.")

    with pytest.raises(ValueError):
        synthesize_sentences(Raising(), "   ")
