"""Transliteration endpoints for language-learning clients."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.transliteration import get_transliteration_service


router = APIRouter(prefix="/api", tags=["transliteration"])


class TransliterateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=15000)
    source: str = "zh"


@router.post("/transliterate")
def transliterate(request: TransliterateRequest):
    """Return local romanization for supported non-Latin languages."""
    try:
        result = get_transliteration_service().transliterate(
            request.text, request.source
        )
        return JSONResponse(content=result.to_dict(), status_code=200)
    except ValueError as exc:
        return JSONResponse(content={"message": str(exc)}, status_code=400)
    except RuntimeError as exc:
        return JSONResponse(content={"message": str(exc)}, status_code=503)
    except Exception as exc:
        return JSONResponse(
            content={"message": f"Transliteration failed: {exc!s}"},
            status_code=502,
        )
