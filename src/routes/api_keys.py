# ruff: noqa: I001
import os
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.constants import (
    SUPPORTED_EMBEDDING_PROVIDERS,
    SUPPORTED_LLM_PROVIDERS,
)
from src.globals import user_dao
from src.utils.path_dependencies import require_user
from src.models.users.api_key import (
    UserAPIKey,
    UserAPIKeyDetailResponse,
    UserAPIKeyResponse,
)
from src.models.users.user import User
from src.utils.crypto_utils import encrypt_str


router = APIRouter(prefix="/api-keys", dependencies=[Depends(require_user)])


LLM_PROVIDER_IDS = {item["id"] for item in SUPPORTED_LLM_PROVIDERS}
EMBEDDING_PROVIDER_IDS = {item["id"] for item in SUPPORTED_EMBEDDING_PROVIDERS}
BOTH_PROVIDER_IDS = LLM_PROVIDER_IDS.intersection(EMBEDDING_PROVIDER_IDS)


def _redact_key(value: str) -> str:
    if len(value) <= 8:
        prefix = value[:4]
        return f"{prefix}****"
    return f"{value[:4]}****{value[-4:]}"


class CreateAPIKeyRequest(BaseModel):
    label: str
    provider: str
    usage: Literal["llm", "embedding", "both"]
    raw_key: str


@router.get("", response_model=list[UserAPIKeyResponse])
def list_api_keys(
    user: Annotated[User, Depends(require_user)],
):
    sorted_keys = sorted(user.api_keys, key=lambda item: item.created_at, reverse=True)
    return [key.to_response() for key in sorted_keys]


@router.post(
    "",
    response_model=UserAPIKeyResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_api_key(
    payload: CreateAPIKeyRequest,
    user: Annotated[User, Depends(require_user)],
):
    label = payload.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Label must not be empty")

    raw_key = payload.raw_key.strip()
    if not raw_key:
        raise HTTPException(status_code=400, detail="API key must not be empty")

    provider = payload.provider.lower().strip()
    usage = payload.usage

    if not provider:
        raise HTTPException(status_code=400, detail="Provider must be specified")

    if usage == "llm" and provider not in LLM_PROVIDER_IDS:
        raise HTTPException(status_code=400, detail="Provider not available for LLM")
    if usage == "embedding" and provider not in EMBEDDING_PROVIDER_IDS:
        raise HTTPException(
            status_code=400, detail="Provider not available for embeddings"
        )
    if usage == "both" and provider not in BOTH_PROVIDER_IDS:
        raise HTTPException(
            status_code=400,
            detail="Provider must support both LLM and embedding usage",
        )

    api_key = UserAPIKey(
        label=label,
        provider=provider,
        usage=usage,
        key_encrypted=encrypt_str(raw_key),
        redacted_key=_redact_key(raw_key),
    )

    user.add_api_key(api_key)
    user_dao.set_user(user)

    return api_key.to_response()


@router.get("/{key_id}", response_model=UserAPIKeyDetailResponse)
def get_api_key_detail(
    key_id: str, user: Annotated[User, Depends(require_user)]
):
    for key in user.api_keys:
        if key.id == key_id:
            return key.to_detail()

    raise HTTPException(status_code=404, detail="API key not found")


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_api_key(
    key_id: str, user: Annotated[User, Depends(require_user)]
):
    # Find and remove the API key
    for i, key in enumerate(user.api_keys):
        if key.id == key_id:
            user.api_keys.pop(i)
            user_dao.set_user(user)
            return
    raise HTTPException(status_code=404, detail="API key not found")
