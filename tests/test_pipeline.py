from src.command import Command, Prompt, prompt_to_json
from src.rag_service.dao import get_database
from src.config import Config
from src.rag_service.embeddings import create_embeddings_model
from src.LLM import LLM, create_llm
from src.pipeline import assemble_prompt
import pytest

use_tokens = False

@pytest.mark.integration
def test_pipeline():
    
    if use_tokens:
            
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
        response = assemble_prompt(command)
        print(response)
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert response != "Error"
        assert response != "No response"
        assert response != "No response found"
        assert response != ""
        assert response == " "
        
    else:
        assert True





