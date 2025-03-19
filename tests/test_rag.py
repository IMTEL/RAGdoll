from fastapi.testclient import TestClient
from src.main import app
from src.rag_service.dao import get_database
import pytest

client = TestClient(app)

# @pytest.mark.integration
# def test_post_rag_item():
#     body = {
#         "text": "Hello from test",
#         "document_id": "docTest123",
#         "document_name": "TestDoc",
#         "NPC": 2,
#         "embedding": [0.5, 0.5]
#     }
#     response = client.post("/rag/post", json=body)
#     assert response.status_code == 200
#     assert response.json()["message"] == "RAG context posted successfully"
