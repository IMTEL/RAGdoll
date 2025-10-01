# test_app.py

import io
import json
import os
from difflib import SequenceMatcher

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from src.main import app
from src.transcribe import transcribe_from_upload


client = TestClient(app)

# Use a small dummy WAV header (not playable, but passes the file check)
DUMMY_WAV = (
    b"RIFF$\x00\x00\x00WAVEfmt "
    + b"\x10\x00\x00\x00\x01\x00\x01\x00"
    + b"\x40\x1f\x00\x00\x80>\x00\x00"
    + b"\x02\x00\x10\x00data\x00\x00\x00\x00"
)

# Sample minimal JSON payload
TEST_JSON = """{
    "action": "test",
    "value": 42
}"""


@pytest.fixture
def dummy_audio_file():
    return io.BytesIO(DUMMY_WAV)


def test_ask_transcribe_with_mock_audio(dummy_audio_file):
    response = client.post(
        "/askTranscribe",
        files={"audio": ("test.wav", dummy_audio_file, "audio/wav")},
        data={"data": (TEST_JSON)},
    )
    assert response.status_code == 200
    # assert "response" in response.json()


def test_ask_transcribe_invalid_json(dummy_audio_file):
    response = client.post(
        "/askTranscribe",
        files={"audio": ("test.wav", dummy_audio_file, "audio/wav")},
        data={"data": "not a json"},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Invalid command."


@pytest.mark.integration
def test_ask_transcribe_with_real_wav():
    file_path = os.path.join("tests", "test_sets", "Chorus.wav")

    with open(file_path, "rb") as audio_file:
        response = client.post(
            "/askTranscribe",
            files={"audio": ("Chorus.wav", audio_file, "audio/wav")},
            data={
                "data": json.dumps(
                    {
                        "scene_name": "ReceptionOutdoor",
                        "user_information": ["User: Tobias", "Mode: Beginner"],
                        "progress": [
                            {
                                "taskName": "Test Task",
                                "description": "A description",
                                "status": "in_progress",
                                "userId": "test_user",
                                "subtaskProgress": [],
                            }
                        ],
                        "user_actions": ["test_action"],
                        "NPC": 123,
                        "chatLog": [{"role": "user", "content": "What is this place?"}],
                    }
                )
            },
        )

    assert response.status_code == 200
    assert "response" in response.json()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


@pytest.mark.integration
def test_transcribe():
    file_path = os.path.join("tests", "test_sets", "Chorus.wav")

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Wrap raw bytes into UploadFile
    upload_file = UploadFile(
        filename="Chorus.wav", file=io.BytesIO(file_bytes), headers="audio/wav"
    )

    result = transcribe_from_upload(upload_file)

    expected = "caught in a landslide no escape from reality open your eyes"

    actual = result

    similarity_score = similarity(expected, actual)

    print(f"Similarity score: {similarity_score:.2f}")
    assert similarity_score > 0.8
