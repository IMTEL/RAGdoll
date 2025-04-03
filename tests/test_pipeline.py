import pytest
from src.command import Command
from src.pipeline import assemble_prompt, create_llm, getAnswerFromUser
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
    """
    Fixture to mock the create_llm function and return a DummyLLM instance.
    """
    # Initialize the response attribute
    DummyLLM.response = None

    def create_mock_llm(model):
        return DummyLLM(DummyLLM.response)

    monkeypatch.setattr("src.pipeline.create_llm", create_mock_llm)
    return create_mock_llm

@pytest.mark.integration
def test_pipeline(mock_llm):
    # Ensure the mock database is used
    db = get_database()
    if not isinstance(db, MockDatabase):
        pytest.skip("Skipping test because MockDatabase is not being used.")

    # Add a document to the mock database with the required keys
    test_document = {
        "text": "This is a test document.",
        "document_name": "test_document",
        "NPC": 100,
        "embedding": [0.1, 0.2, 0.3],
        "document_id": "test_id"
    }
    db.post_context(
        text=test_document["text"],
        document_name=test_document["document_name"],
        NPC=test_document["NPC"],
        embedding=test_document["embedding"],
        document_id=test_document["document_id"],
    )

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

    # Set the response for the mock LLM
    DummyLLM.response = "This is a mock response."

    response = assemble_prompt(command, "mock")
    print(response)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert response != "Error"
    assert response != "No response"
    assert response != "No response found"
    assert response != ""

def test_valid_response_name(monkeypatch):
    # Set the response for the mock LLM
    DummyLLM.response = 'name: "John Doe"'
    def create_mock_llm(model):
        return DummyLLM(DummyLLM.response)
    monkeypatch.setattr("src.pipeline.create_llm", create_mock_llm)
    answer = "My name is John Doe"
    target = "name"
    question = "What is your name?"
    result = getAnswerFromUser(answer, target, question)
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
    result = getAnswerFromUser(answer, target, question)
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
    result = getAnswerFromUser(answer, target, question)
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
    result = getAnswerFromUser(answer, target, question)
    assert result == "Empty response from the language model."
