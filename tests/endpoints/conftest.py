"""Shared fixtures for endpoint tests that need real authorization wiring.

The app's services in ``src.globals`` are built at import time against the
configured backends (Mongo, Keycloak). These fixtures swap them for the real
``AccessService`` and ``AuthService`` backed by in-memory mocks, so the
authorization logic under test is the production code path.
"""

import base64
import json
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from src.access_service.access_service import AccessService
from src.auth.auth_service.auth_service import AuthService
from src.main import app
from src.models.agent import Agent
from src.models.users.user import User
from src.routes import agents as agents_routes
from src.routes import chat as chat_routes
from src.utils import path_dependencies
from tests.mocks.mock_agent_dao import MockAgentDAO
from tests.mocks.mock_auth_provider import MockAuthProvider
from tests.mocks.mock_user_dao import MockUserDao


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def make_jwt(subject: str) -> str:
    """Build a JWT-shaped token (unsigned) whose subject is ``subject``."""
    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({"sub": subject}).encode())
    return f"{header}.{payload}.{_b64url(b'signature')}"


# A JWT-shaped token the fake identity provider rejects (like an expired one).
INVALID_JWT = make_jwt("expired-user")


class _FakeIdentityProvider(MockAuthProvider):
    """MockAuthProvider that rejects ``INVALID_JWT`` the way PyJWT would."""

    def get_authenticated_user(self, token: str) -> User | None:
        if token == INVALID_JWT:
            raise ValueError("Signature has expired")
        return super().get_authenticated_user(token)


def sample_agent(name: str = "Test Agent") -> Agent:
    return Agent(
        name=name,
        description="Agent used by endpoint tests",
        prompt="You are a test assistant",
        llm_model="gpt-4",
        llm_api_key="sk-secret-llm-key",
        embedding_model="text-embedding-ada-002",
        embedding_api_key="secret-embedding-key",
        last_updated="2026-10-18T00:00:00Z",
    )


@dataclass
class AuthEnv:
    client: TestClient
    agent_dao: MockAgentDAO
    user_dao: MockUserDao
    access_service: AccessService

    def add_user(self, subject: str, **fields) -> tuple[User, str]:
        """Register a user that the fake provider resolves from the returned token."""
        token = make_jwt(subject)
        user = self.user_dao.set_user(
            User(
                auth_provider=MockAuthProvider.get_provider(),
                provider_user_id=token,
                **fields,
            )
        )
        return user, token


@pytest.fixture
def auth_env(monkeypatch) -> AuthEnv:
    agent_dao = MockAgentDAO()
    agent_dao.clear()
    user_dao = MockUserDao()
    access_service = AccessService(agent_dao)
    auth_service = AuthService(
        user_db=user_dao,
        auth_provider_factory=lambda _provider, db: _FakeIdentityProvider(db),
    )

    for module in (path_dependencies, agents_routes):
        monkeypatch.setattr(module, "access_service", access_service)
        monkeypatch.setattr(module, "agent_dao", agent_dao)
    monkeypatch.setattr(path_dependencies, "auth_service", auth_service)
    monkeypatch.setattr(agents_routes, "user_dao", user_dao)
    # Never call an LLM from these tests.
    monkeypatch.setattr(
        chat_routes,
        "assemble_prompt_with_agent",
        lambda command, agent: {"response": f"answer from {agent.id}"},
    )

    yield AuthEnv(
        client=TestClient(app),
        agent_dao=agent_dao,
        user_dao=user_dao,
        access_service=access_service,
    )
    agent_dao.clear()
