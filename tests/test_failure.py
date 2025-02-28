from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_receive_failure():
    """
    Test if /api/failure endpoint receives data and stores it in memory.
    """

    # Some failure data to send
    failure_data = {
        "errorCode": "404",
        "description": "Task not found"
    }

    # POST request to /api/failure
    response = client.post("/api/failure", json=failure_data)

    # Check 200 OK status code
    assert response.status_code == 200

    # Check correct response message
    response_json = response.json()
    assert response_json["message"] == "Failure received successfully"
    assert response_json["data"]["errorCode"] == "404"
    assert response_json["data"]["description"] == "Task not found"
    assert "receivedAt" in response_json["data"]
