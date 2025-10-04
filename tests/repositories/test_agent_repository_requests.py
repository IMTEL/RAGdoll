"""Integration tests for HTTP endpoints using repositories.

Tests the FastAPI endpoints for both agent and context repositories
to ensure proper HTTP request/response handling.
"""

import os


import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.rag_service.repositories import get_agent_repository
from tests.mocks import MockAgentRepository


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_mock_repositories():
    """Clear all mock repositories before and after each test."""
    repo = get_agent_repository()
    if isinstance(repo, MockAgentRepository):
        repo.clear()
    yield
    if isinstance(repo, MockAgentRepository):
        repo.clear()


class TestAgentEndpoints:
    """Tests for agent-related HTTP endpoints."""

    def test_create_agent_success(self):
        """Test creating an agent via POST /agents/."""
        agent_data = {
            "name": "Test Assistant",
            "description": "An assistant for testing",
            "prompt": "You are a helpful test assistant",
            "corpa": ["doc1", "doc2"],
            "roles": [
                {
                    "name": "user",
                    "description": "Standard user",
                    "subset_of_corpa": [0, 1],
                }
            ],
            "llm_model": "gpt-3.5-turbo",
            "llm_temperature": 0.7,
            "llm_max_tokens": 1000,
            "llm_api_key": "sk-test123",
            "access_key": ["key1"],
            "retrieval_method": "semantic",
            "embedding_model": "text-embedding-ada-002",
            "status": "active",
            "response_format": "json",
            "last_updated": "2025-10-03T10:00:00Z",
        }

        response = client.post("/agents/", json=agent_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Assistant"
        assert data["description"] == "An assistant for testing"
        assert data["llm_model"] == "gpt-3.5-turbo"

    def test_create_agent_missing_field(self):
        """Test creating an agent with missing required fields."""
        incomplete_agent = {
            "name": "Incomplete Agent",
            # Missing many required fields
        }

        response = client.post("/agents/", json=incomplete_agent)

        assert response.status_code == 422  # Validation error

    def test_get_agents_empty(self):
        """Test getting agents when repository is empty."""
        response = client.get("/agents/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_get_agents_multiple(self):
        """Test retrieving multiple agents."""
        # Create first agent
        agent1_data = {
            "name": "Agent One",
            "description": "First agent",
            "prompt": "Prompt 1",
            "corpa": ["doc1"],
            "roles": [
                {"name": "user", "description": "User role", "subset_of_corpa": [0]}
            ],
            "llm_model": "gpt-3.5-turbo",
            "llm_temperature": 0.5,
            "llm_max_tokens": 500,
            "llm_api_key": "sk-key1",
            "access_key": ["key1"],
            "retrieval_method": "semantic",
            "embedding_model": "text-embedding-ada-002",
            "status": "active",
            "response_format": "text",
            "last_updated": "2025-10-03T10:00:00Z",
        }

        # Create second agent
        agent2_data = {
            "name": "Agent Two",
            "description": "Second agent",
            "prompt": "Prompt 2",
            "corpa": ["doc2"],
            "roles": [
                {"name": "admin", "description": "Admin role", "subset_of_corpa": [0]}
            ],
            "llm_model": "gpt-4",
            "llm_temperature": 0.7,
            "llm_max_tokens": 1000,
            "llm_api_key": "sk-key2",
            "access_key": ["key2"],
            "retrieval_method": "keyword",
            "embedding_model": "text-embedding-ada-002",
            "status": "inactive",
            "response_format": "json",
            "last_updated": "2025-10-03T11:00:00Z",
        }

        # Create both agents
        client.post("/agents/", json=agent1_data)
        client.post("/agents/", json=agent2_data)

        # Get all agents
        response = client.get("/agents/")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "Agent One"
        assert data[1]["name"] == "Agent Two"

    def test_get_agent_by_id_success(self):
        """Test retrieving a specific agent by ID."""
        # Create an agent first
        agent_data = {
            "name": "Specific Agent",
            "description": "Agent to retrieve by ID",
            "prompt": "Test prompt",
            "corpa": ["doc1"],
            "roles": [{"name": "user", "description": "User", "subset_of_corpa": [0]}],
            "llm_model": "gpt-3.5-turbo",
            "llm_temperature": 0.7,
            "llm_max_tokens": 1000,
            "llm_api_key": "sk-test",
            "access_key": ["key1"],
            "retrieval_method": "semantic",
            "embedding_model": "text-embedding-ada-002",
            "status": "active",
            "response_format": "json",
            "last_updated": "2025-10-03T10:00:00Z",
        }

        # Create the agent
        create_response = client.post("/agents/", json=agent_data)
        assert create_response.status_code == 200

        # Since repository is cleared before each test, this should be the only agent (ID = 0)
        response = client.get("/agents/0")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Specific Agent"

    def test_get_agent_by_id_not_found(self):
        """Test retrieving non-existent agent returns 404."""
        response = client.get("/agents/999")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_agent_data_integrity(self):
        """Test that agent data is preserved through create and retrieve."""
        agent_data = {
            "name": "Integrity Test Agent",
            "description": "Testing data integrity",
            "prompt": "You are testing data integrity",
            "corpa": ["corpus1", "corpus2", "corpus3"],
            "roles": [
                {
                    "name": "admin",
                    "description": "Admin role",
                    "subset_of_corpa": [0, 1, 2],
                },
                {"name": "user", "description": "User role", "subset_of_corpa": [1]},
            ],
            "llm_model": "gpt-4",
            "llm_temperature": 0.8,
            "llm_max_tokens": 2000,
            "llm_api_key": "sk-integrity-test",
            "access_key": ["key1", "key2", "key3"],
            "retrieval_method": "hybrid",
            "embedding_model": "text-embedding-ada-002",
            "status": "testing",
            "response_format": "markdown",
            "last_updated": "2025-10-03T12:30:00Z",
        }

        # Create agent
        create_response = client.post("/agents/", json=agent_data)
        assert create_response.status_code == 200

        # Retrieve all agents and find the one we just created
        get_response = client.get("/agents/")
        assert get_response.status_code == 200

        # Find the agent we just created by name (should be the only one if fixture cleared)
        agents = get_response.json()
        retrieved = next((a for a in agents if a["name"] == agent_data["name"]), None)
        assert retrieved is not None, (
            f"Could not find agent with name {agent_data['name']}"
        )

        # Verify all fields match
        assert retrieved["name"] == agent_data["name"]
        assert retrieved["description"] == agent_data["description"]
        assert retrieved["prompt"] == agent_data["prompt"]
        assert retrieved["corpa"] == agent_data["corpa"]
        assert len(retrieved["roles"]) == 2
        assert retrieved["llm_model"] == agent_data["llm_model"]
        assert retrieved["llm_temperature"] == agent_data["llm_temperature"]
        assert retrieved["llm_max_tokens"] == agent_data["llm_max_tokens"]
        assert retrieved["retrieval_method"] == agent_data["retrieval_method"]
        assert retrieved["status"] == agent_data["status"]

    def test_invalid_agent_id_format(self):
        """Test that invalid ID format returns 404."""
        response = client.get("/agents/invalid-id-format")

        assert response.status_code == 404


class TestRepositoryHealthCheck:
    """Tests for repository health and connectivity."""

    def test_agent_repository_reachable(self):
        """Test that agent repository is reachable."""
        repo = get_agent_repository()
        assert repo.is_reachable() is True

    def test_repositories_initialized(self):
        """Test that repositories are properly initialized."""
        agent_repo = get_agent_repository()

        assert agent_repo is not None
        assert hasattr(agent_repo, "create_agent")
        assert hasattr(agent_repo, "get_agents")
        assert hasattr(agent_repo, "get_agent_by_id")
