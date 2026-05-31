import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.globals import agent_dao, user_dao
from src.models.users.user import User
from src.utils.global_logs import failure_log, progress_log


router = APIRouter()


class BootstrapKeycloakUserRequest(BaseModel):
    provider_user_id: str = Field(..., min_length=1)
    email: str | None = None
    name: str | None = None
    attach_all_agents: bool = True
    migrate_from_providers: list[str] = Field(
        default_factory=lambda: ["demo", "mock", "dev"]
    )


@router.get("/api/debug/logs")
def get_logs():
    """Returns the in-memory logs for progress and failure."""
    return {"progressLog": progress_log, "failureLog": failure_log}


@router.post("/api/debug/bootstrap-keycloak-user")
def bootstrap_keycloak_user(payload: BootstrapKeycloakUserRequest):
    """Local migration helper for attaching existing data to a Keycloak user.

    This endpoint is intentionally disabled unless AUTH_BOOTSTRAP_ENABLED=true.
    It is useful when moving from DISABLE_AUTH/demo mode to Keycloak locally.
    """
    if os.getenv("AUTH_BOOTSTRAP_ENABLED", "").lower() != "true":
        raise HTTPException(status_code=404, detail="Not found")

    user = user_dao.get_user_by_provider("keycloak", payload.provider_user_id)
    if user is None:
        user = User(
            auth_provider="keycloak",
            provider_user_id=payload.provider_user_id,
            email=payload.email,
            name=payload.name,
            owned_agents=[],
            api_keys=[],
        )
    else:
        user.email = payload.email or user.email
        user.name = payload.name or user.name

    owned_agents = set(user.owned_agents)
    if payload.attach_all_agents:
        owned_agents.update(
            agent.id for agent in agent_dao.get_agents() if agent.id is not None
        )

    existing_api_key_ids = {key.id for key in user.api_keys}
    for provider in payload.migrate_from_providers:
        provider_user_ids = [provider]
        if provider == "dev":
            provider_user_ids.append("dev-provider-id")

        source_user = None
        for provider_user_id in provider_user_ids:
            source_user = user_dao.get_user_by_provider(provider, provider_user_id)
            if source_user is not None:
                break
        if source_user is None:
            continue

        owned_agents.update(source_user.owned_agents)
        for api_key in source_user.api_keys:
            if api_key.id not in existing_api_key_ids:
                user.api_keys.append(api_key)
                existing_api_key_ids.add(api_key.id)

    user.owned_agents = sorted(owned_agents)
    user = user_dao.set_user(user)
    return {
        "id": user.id,
        "auth_provider": user.auth_provider,
        "provider_user_id": user.provider_user_id,
        "owned_agents": user.owned_agents,
        "api_key_count": len(user.api_keys),
    }
