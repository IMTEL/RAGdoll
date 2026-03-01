# ruff: noqa: I001
import os
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Request
from fastapi_jwt_auth import AuthJWT

router = APIRouter()


def optional_auth(request: Request) -> Optional[AuthJWT]:
    """Return AuthJWT only if auth is enabled and header is present."""
    if os.getenv("DISABLE_AUTH", "").lower() == "true":
        return None
    return AuthJWT(request)


@router.get("/providers")
def get_supported_providers(
    authorize: Annotated[Optional[AuthJWT], Depends(optional_auth)] = None,
):
    if os.getenv("DISABLE_AUTH", "").lower() != "true" and authorize is not None:
        authorize.jwt_required()
    from src.constants import (
        SUPPORTED_EMBEDDING_PROVIDERS,
        SUPPORTED_LLM_PROVIDERS,
    )

    return {
        "llm": SUPPORTED_LLM_PROVIDERS,
        "embedding": SUPPORTED_EMBEDDING_PROVIDERS,
    }
