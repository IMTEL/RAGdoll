from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_receive_progress():
    """
    Test if /api/progress endpoint receives data and stores it in memory.
    """

    # Some progress data to send
    progress_data = {
        "taskName": "TestTask",
        "status": "complete"
    }

    # POST
    response = client.post("/api/progress", json=progress_data)

    # Check 200 OK
    assert response.status_code == 200

    # Check correct response
    response_json = response.json()
    assert response_json["message"] == "Progress received successfully"
    assert response_json["data"]["taskName"] == "TestTask"
    assert response_json["data"]["status"] == "complete"
