from src.command import Command, Prompt, prompt_to_json
from src.rag_service.dao import get_database, MockDatabase
from src.config import Config
from src.rag_service.embeddings import create_embeddings_model
from src.LLM import LLM
from src.pipeline import assemble_prompt, getAnswerFromUser
import pytest

use_tokens = False

@pytest.mark.integration
def test_pipeline():
    # Ensure the mock database is used
    db = get_database()
    if not isinstance(db, MockDatabase):
        pytest.skip("Skipping test because MockDatabase is not being used.")

    # Add a document to the mock database with the required keys
    test_document = {
        "text": "This is a test document.",
        "documentName": "test_document",  # Include 'documentName'
        "NPC": 100,
        "embedding": [0.1, 0.2, 0.3],
        "documentId": "test_id"
    }
    db.post_context(**test_document)

    command = Command(
        user_name="Tobias",
        user_mode="Used to VR, but dont know the game",
        question="Why does salmon swim upstream?",
        progress="""{
            "taskName": "Daily Exercise Routine",
            "description": "Complete daily fitness routine to improve overall health",
            "status": "start",
            "userId": "user123",
            "subtaskProgress": [
                {
                    "subtaskName": "Warm Up",
                    "description": "Prepare muscles for workout",
                    "completed": False,
                    "stepProgress": [
                        {
                            "stepName": "Jumping Jacks",
                            "repetitionNumber": 30,
                            "completed": False
                        },
                        {
                            "stepName": "Arm Circles",
                            "repetitionNumber": 20,
                            "completed": False
                        }
                        ]
                    }
                ]
            }""",
        user_actions=["Not implemented"],
        NPC=100
    )
    response = assemble_prompt(command, "mock")
    print(response)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert response != "Error"
    assert response != "No response"
    assert response != "No response found"
    assert response != ""
 
 
import pytest
import pipeline  # Import the module where getAnswerFromUser is defined

# Dummy language model for testing.
class DummyLLM:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return self.response

# Dummy create_llm function that returns our DummyLLM.
def dummy_create_llm(model):
    return DummyLLM(dummy_create_llm.response)

# Use pytest's monkeypatch to replace pipeline.create_llm with our dummy_create_llm.
@pytest.fixture(autouse=True)
def patch_create_llm(monkeypatch):
    monkeypatch.setattr(pipeline, "create_llm", dummy_create_llm)

def test_valid_response_name():
    dummy_create_llm.response = 'name: "John Doe"'
    answer = "My name is John Doe"
    target = "name"
    question = "What is your name?"
    result = pipeline.getAnswerFromUser(answer, target, question)
    assert result == 'name: "John Doe"'

def test_valid_response_user_mode():
    dummy_create_llm.response = 'user_mode: "beginner"'
    answer = "I am not experienced with VR"
    target = "user_mode"
    question = "How do you rate your VR experience?"
    result = pipeline.getAnswerFromUser(answer, target, question)
    assert result == 'user_mode: "beginner"'

def test_none_response():
    dummy_create_llm.response = None
    answer = "I am not experienced with VR"
    target = "user_mode"
    question = "How do you rate your VR experience?"
    result = pipeline.getAnswerFromUser(answer, target, question)
    assert result == "No response from the language model."

def test_empty_response():
    dummy_create_llm.response = ""
    answer = "I am not experienced with VR"
    target = "user_mode"
    question = "How do you rate your VR experience?"
    result = pipeline.getAnswerFromUser(answer, target, question)
    assert result == "Empty response from the language model."
