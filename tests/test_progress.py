from fastapi.testclient import TestClient
from src.main import app
from datetime import datetime, timezone
from src.utils.global_logs import progressLog  # Import the global log

client = TestClient(app)

class TestProgress:
    def setup_method(self):
        """
        Setup before each test: Clear the progress log to ensure no state leakage.
        """
        progressLog.clear()

    def teardown_method(self):
        """
        Teardown after each test: Clear the progress log to maintain test isolation.
        """
        progressLog.clear()

    def test_receive_progress(self):
        """
        Test if /api/progress endpoint receives data and stores it in memory.
        """

        # Send a "start" status first
        start_data = {
            "taskName": "TestTask",
            "status": "start"
        }
        start_response = client.post("/api/progress", json=start_data)
        assert start_response.status_code == 200

        # Check correct response
        response_json = start_response.json()
        assert response_json["message"] == "Progress received successfully"
        assert response_json["data"]["taskName"] == "TestTask"
        assert response_json["data"]["status"] == "start"
        assert "startedAt" in response_json["data"]
        assert response_json["data"]["completedAt"] is None

    def test_receive_progress_complete(self):
        """
        Test if /api/progress sets completedAt when status is 'complete'.
        """

        # 1. Send a "start" status first to initialize the task
        start_data = {
            "taskName": "TestTask",
            "status": "start"
        }
        start_response = client.post("/api/progress", json=start_data)
        assert start_response.status_code == 200

        # 2. Now send a "complete" status
        complete_data = {
            "taskName": "TestTask",
            "status": "complete"
        }
        complete_response = client.post("/api/progress", json=complete_data)
        assert complete_response.status_code == 200

        response_json = complete_response.json()
        assert response_json["message"] == "Progress received successfully"
        assert response_json["data"]["taskName"] == "TestTask"
        assert response_json["data"]["status"] == "complete"
        assert "startedAt" in response_json["data"]
        assert response_json["data"]["completedAt"] is not None

        # Check that the timestamp is in UTC
        completed_at = datetime.fromisoformat(response_json["data"]["completedAt"])
        assert completed_at.tzinfo == timezone.utc
