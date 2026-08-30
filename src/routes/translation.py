"""Translation endpoints for external language-learning clients."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.translation import get_translation_service


router = APIRouter(prefix="/api", tags=["translation"])


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=15000)
    source: str = "auto"
    target: str = "en"


class BatchTranslateRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    source: str = "auto"
    target: str = "en"


@router.post("/translate")
def translate(request: TranslateRequest):
    """Translate text using the configured backend translation provider."""
    try:
        result = get_translation_service().translate(
            request.text.strip(),
            source=request.source.strip() or "auto",
            target=request.target.strip() or "en",
        )
        return JSONResponse(content=result.to_dict(), status_code=200)
    except RuntimeError as exc:
        return JSONResponse(content={"message": str(exc)}, status_code=503)
    except Exception as exc:
        return JSONResponse(
            content={"message": f"Translation failed: {exc!s}"},
            status_code=502,
        )


@router.post("/translate/batch")
def translate_batch(request: BatchTranslateRequest):
    """Translate multiple texts with one client request."""
    texts = [text.strip() for text in request.texts if text.strip()]
    if not texts:
        return JSONResponse(content={"message": "At least one text is required."}, status_code=400)

    try:
        results = get_translation_service().translate_many(
            texts,
            source=request.source.strip() or "auto",
            target=request.target.strip() or "en",
        )
        return JSONResponse(
            content={"translations": [result.to_dict() for result in results]},
            status_code=200,
        )
    except RuntimeError as exc:
        return JSONResponse(content={"message": str(exc)}, status_code=503)
    except Exception as exc:
        return JSONResponse(
            content={"message": f"Batch translation failed: {exc!s}"},
            status_code=502,
        )


@router.post("/translate/warmup")
def warmup_translation(source: str = "es", target: str = "en"):
    """Warm LibreTranslate and its local language model."""
    try:
        return JSONResponse(
            content=get_translation_service().warmup(source=source, target=target),
            status_code=200,
        )
    except RuntimeError as exc:
        return JSONResponse(content={"message": str(exc)}, status_code=503)
    except Exception as exc:
        return JSONResponse(
            content={"message": f"Translation warmup failed: {exc!s}"},
            status_code=502,
        )
