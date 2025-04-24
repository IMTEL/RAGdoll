import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from src.web_socket_sandbox import app, ws_manager
import json

# Synchronous client fixture for regular tests
@pytest.fixture
def client():
    return TestClient(app)

# Async client fixture for async tests
@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def clean_ws_manager():
    # Clean up the WebSocketManager before each test
    ws_manager.active_connections = {}
    yield ws_manager
    ws_manager.active_connections = {}  # Clean up after test

def test_websocket_connection(client, clean_ws_manager):
    with client.websocket_connect("/ws/test_session") as websocket:
        assert "test_session" in clean_ws_manager.active_connections
        assert len(clean_ws_manager.active_connections) == 1

def test_websocket_disconnect(client, clean_ws_manager):
    with client.websocket_connect("/ws/test_session") as websocket:
        pass  # Connection closes when exiting context
    
    assert "test_session" not in clean_ws_manager.active_connections
    assert len(clean_ws_manager.active_connections) == 0

def test_multiple_connections(client, clean_ws_manager):
    with client.websocket_connect("/ws/session1") as ws1, \
         client.websocket_connect("/ws/session2") as ws2:
        
        assert len(clean_ws_manager.active_connections) == 2
        assert "session1" in clean_ws_manager.active_connections
        assert "session2" in clean_ws_manager.active_connections

@pytest.mark.asyncio
async def test_send_message(async_client, clean_ws_manager):
    async with async_client.websocket_connect("/ws/test_session") as websocket:
        test_message = "Hello, WebSocket!"
        await clean_ws_manager.send_message("test_session", test_message)
        data = await websocket.receive_text()
        assert data == test_message

@pytest.mark.asyncio
async def test_invalid_session_message(async_client, clean_ws_manager):
    async with async_client.websocket_connect("/ws/test_session") as websocket:
        # This should not raise an error
        await clean_ws_manager.send_message("nonexistent_session", "Hello")
        
        # Verify connection is still alive
        await websocket.send_text("ping")
        assert await websocket.receive_text() == "ping"

@pytest.mark.asyncio
async def test_user_prompt_event(async_client, clean_ws_manager, mocker):
    mock_llm = mocker.MagicMock()
    mock_llm.generate.return_value = "Mocked AI response"
    mocker.patch("src.web_socket_sandbox.create_llm", return_value=mock_llm)
    
    test_prompt = "What is the meaning of life?"
    
    async with async_client.websocket_connect("/ws/test_session") as websocket:
        message = {
            "event": "user_prompt",
            "data": {
                "prompt": test_prompt,
                "conversation_id": "conv123"
            }
        }
        await websocket.send_json(message)
        
        # Give some time for processing
        try:
            await websocket.receive_text(timeout=1.0)
        except Exception:
            pass  # We don't actually expect a response in this mock
        
        mock_llm.generate.assert_called_once_with(test_prompt)

def test_unknown_event(client, clean_ws_manager):
    with client.websocket_connect("/ws/test_session") as websocket:
        websocket.send_json({
            "event": "unknown_event",
            "data": {}
        })
        # Verify connection is still alive
        websocket.send_text("ping")
        assert websocket.receive_text() == "ping"

def test_invalid_json(client, clean_ws_manager):
    with client.websocket_connect("/ws/test_session") as websocket:
        websocket.send_text("This is not JSON")
        # Verify connection is still alive
        websocket.send_text("ping")
        assert websocket.receive_text() == "ping"