from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.llm import list_llm_models
from src.models.errors.embedding_error import EmbeddingAPIError, EmbeddingError
from src.models.errors.llm_error import LLMAPIError
from src.models.model import Model
from src.models.users.user import User
from src.rag_service.embeddings import list_embedding_models
from src.routes.agents import ProviderKeyRequest
from src.utils.crypto_utils import decrypt_value
from src.utils.path_dependencies import require_user


router = APIRouter(tags=["Providers"])


def _map_embedding_api_error(error: EmbeddingAPIError) -> int:
    message = str(error.original_error).lower() if error.original_error else ""
    if any(keyword in message for keyword in ("quota", "rate limit", "429")):
        return 429
    if any(
        keyword in message
        for keyword in (
            "unauthorized",
            "forbidden",
            "authentication",
            "api key",
            "permission",
        )
    ):
        return 401
    return 400


@router.get("/providers", dependencies=[Depends(require_user)])
def get_supported_providers():
    from src.constants import (
        SUPPORTED_EMBEDDING_PROVIDERS,
        SUPPORTED_LLM_PROVIDERS,
    )

    return {
        "llm": SUPPORTED_LLM_PROVIDERS,
        "embedding": SUPPORTED_EMBEDDING_PROVIDERS,
    }


@router.post(
    "/get_models",
    response_model=list[Model],
    dependencies=[Depends(require_user)],
    deprecated=True,
    responses={400: {"description": "Provider does not exist"}},
)
def fetch_models(
    payload: ProviderKeyRequest,
):
    try:
        return list_llm_models(payload.provider, payload.api_key)
    except LLMAPIError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get(
    "/models/llm/{api_key_id}",
    dependencies=[Depends(require_user)],
    responses={
        400: {"description": "if the llm value can not be found"},
        404: {"description": "If the key can not be found within the account"},
    },
)
def fetch_llm_models_with_api(
    api_key_id: str, user: Annotated[User, Depends(require_user)]
) -> list[Model]:
    """Fetches the models accessible with a apiKey.

    Args:
        api_key_id (str): The id of thee api key
        user (Annotated[User, Depends): The user fetched from the authorization header

    Raises:
        HTTPException: Forward of the LLM API error
        HTTPException: 400, the value that is incorrect
        HTTPException: 404, if the key can not be found within the user

    Returns:
        list[Model]: the models that can be fetched from the provider with the api key
    """
    for api in user.api_keys:
        if api.id == api_key_id:
            try:
                return list_llm_models(api.provider, decrypt_value(api.key_encrypted))
            except LLMAPIError as error:
                raise HTTPException(
                    status_code=error.status_code, detail=str(error)
                ) from error
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
    raise HTTPException(status_code=404, detail="Key not found")


@router.post(
    "/get_embedding_models",
    response_model=list[str],
    dependencies=[Depends(require_user)],
)
def fetch_embedding_models(
    payload: ProviderKeyRequest,
):
    try:
        return list_embedding_models(payload.provider, payload.api_key)
    except EmbeddingAPIError as error:
        status_code = _map_embedding_api_error(error)
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    except EmbeddingError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
