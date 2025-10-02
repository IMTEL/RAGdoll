import pytest

from src.command import Command
from src.models.message import Message
from src.models.progress import ProgressData
from src.pipeline import assemble_prompt, get_answer_from_user
from src.rag_service.dao import MockDatabase, get_database


# Dummy language model for testing.
class DummyLLM:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return self.response


# Dummy create_llm function that returns our DummyLLM.
def dummy_create_llm(model):
    return DummyLLM(dummy_create_llm.response)


@pytest.fixture
def mock_llm(monkeypatch):
    """Fixture to mock the create_llm function and return a DummyLLM instance."""
    # Initialize the response attribute
    DummyLLM.response = None

    def create_mock_llm(model):
        return DummyLLM(DummyLLM.response)

    monkeypatch.setattr("src.pipeline.create_llm", create_mock_llm)
    return create_mock_llm


@pytest.mark.integration
def test_pipeline(mock_llm):
    db = get_database()
    if not isinstance(db, MockDatabase):
        pytest.skip("Skipping test because MockDatabase is not being used.")

    test_document = {
        "text": "This is a test document.",
        "document_name": "test_document",
        "category": "Miscellaneous",
        "embedding": [0.1, 0.2, 0.3],
        "document_id": "test_id",
    }
    db.post_context(**test_document)

    progress = [
        ProgressData(
            task_name="Daily Exercise Routine",
            description="Improve health",
            status="start",
            user_id="user123",
            subtask_progress=[],
        )
    ]

    command = Command(
        scene_name="Laboratory",
        user_information=["Name: Tobias", "Mode: Beginner"],
        progress=progress,
        user_actions=["Not implemented"],
        npc=100,
        chat_log=[Message(role="user", content="Why does salmon swim upstream?")],
    )

    # Set the response for the mock LLM
    DummyLLM.response = "This is a mock response."
    response = assemble_prompt(command, "mock")
    assert response is not None
    assert len(response) > 0
    assert isinstance(response, dict) and "choices" in response


def test_valid_response_name(monkeypatch):
    # Set the response for the mock LLM
    DummyLLM.response = 'name: "John Doe"'

    def create_mock_llm(model):
        return DummyLLM(DummyLLM.response)

    monkeypatch.setattr("src.pipeline.create_llm", create_mock_llm)
    answer = "My name is John Doe"
    target = "name"
    question = "What is your name?"
    result = get_answer_from_user(answer, target, question)
    assert result == 'name: "John Doe"'


def test_valid_response_user_mode(monkeypatch):
    # Set the response for the mock LLM
    DummyLLM.response = 'user_mode: "beginner"'

    def create_mock_llm(model):
        return DummyLLM(DummyLLM.response)

    monkeypatch.setattr("src.pipeline.create_llm", create_mock_llm)
    answer = "I am not experienced with VR"
    target = "user_mode"
    question = "How do you rate your VR experience?"
    result = get_answer_from_user(answer, target, question)
    assert result == 'user_mode: "beginner"'


def test_none_response(monkeypatch):
    # Set the response for the mock LLM
    DummyLLM.response = None

    def create_mock_llm(model):
        return DummyLLM(DummyLLM.response)

    monkeypatch.setattr("src.pipeline.create_llm", create_mock_llm)
    answer = "I am not experienced with VR"
    target = "user_mode"
    question = "How do you rate your VR experience?"
    result = get_answer_from_user(answer, target, question)
    assert result == "No response from the language model."


def test_empty_response(monkeypatch):
    # Set the response for the mock LLM
    DummyLLM.response = ""

    def create_mock_llm(model):
        return DummyLLM(DummyLLM.response)

    monkeypatch.setattr("src.pipeline.create_llm", create_mock_llm)
    answer = "I am not experienced with VR"
    target = "user_mode"
    question = "How do you rate your VR experience?"
    result = get_answer_from_user(answer, target, question)
    assert result == "Empty response from the language model."
