# ruff: noqa: I001
import os
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request


router = APIRouter()


def optional_auth(request: Request) -> str | None:
    """Return AuthJWT only if auth is enabled and header is present."""

    auth_header = request.headers.get("authorization")
    if not auth_header:
        # If no header and auth is required, AuthJWT will handle the error
        raise HTTPException(status_code=403, detail="Unauthorized")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=400, detail="Invalid header bearer token")
    return auth_header[7:]


@router.get("/providers")
def get_supported_providers(
    authorize: Annotated[Optional[str], Depends(optional_auth)] = None,
):
    from src.constants import (
        SUPPORTED_EMBEDDING_PROVIDERS,
        SUPPORTED_LLM_PROVIDERS,
    )

    return {
        "llm": SUPPORTED_LLM_PROVIDERS,
        "embedding": SUPPORTED_EMBEDDING_PROVIDERS,
    }
