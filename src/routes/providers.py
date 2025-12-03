# ruff: noqa: I001
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi_jwt_auth import AuthJWT

router = APIRouter()


@router.get("/providers")
def get_supported_providers(authorize: Annotated[AuthJWT, Depends()] = None):
    # # authorize.jwt_required()
    from src.constants import (
        SUPPORTED_EMBEDDING_PROVIDERS,
        SUPPORTED_LLM_PROVIDERS,
    )

    return {
        "llm": SUPPORTED_LLM_PROVIDERS,
        "embedding": SUPPORTED_EMBEDDING_PROVIDERS,
    }
