import logging
from contextlib import asynccontextmanager
from logging.config import dictConfig

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jwt.exceptions import PyJWTError

from src.config import Config
from src.constants import LOGGING_CONFIG
from src.routes import (
    agents,
    api_keys,
    chat,
    debug,
    feedback,
    profiles,
    progress,
    providers,
    templates,
    translation,
    transliteration,
    upload,
    user_settings,
)


# Apply the logging configuration
dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)
config = Config()


def warn_if_speech_endpoints_public(public: bool) -> None:
    """Log a warning when the speech and translation routes need no credentials."""
    if public:
        logger.warning(
            "PUBLIC_SPEECH_ENDPOINTS=true: /api/chat/transcribe, /api/chat/tts, "
            "/api/chat/tts/warmup, /api/chat/stt/warmup, /api/translate* and "
            "/api/transliterate* accept requests without credentials."
        )


warn_if_speech_endpoints_public(config.PUBLIC_SPEECH_ENDPOINTS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Remove upload temp files left behind by an earlier crash."""
    upload.sweep_stale_temp_files()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Chat-Service Microservice API",
    description="Generate prompts with context and passing them to LLM.",
    root_path=config.PROXY_PATH,
    version="1.0.0",
)

# CORS middleware
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        config.RAGDOLL_CONFIG_API_URL,
        config.RAGDOLL_CHAT_API_URL,
        *config.CORS_ALLOWED_ORIGINS,
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # Allow all headers
)


# Progress router
app.include_router(progress.router)

# Debug router
app.include_router(debug.router)

# Upload router
app.include_router(upload.router)

# Agents router
app.include_router(agents.router)

# API Keys router
app.include_router(api_keys.router)

# Templates router
app.include_router(templates.router)

# User settings router
app.include_router(user_settings.router)

# Provider router
app.include_router(providers.router)

# Chat router (handles /ask, /transcribe, /askTranscribe)
app.include_router(chat.router)

# Translation router
app.include_router(translation.router)

# Transliteration router
app.include_router(transliteration.router)

# Job seeker profiles router (TDT4290)
app.include_router(profiles.router)

# Message feedback router (TDT4290)
app.include_router(feedback.router)


@app.get("/")
def hello_world():
    """Simple root endpoint to verify service is running.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the RAGdoll backend!"}


@app.get("/ping")
def ping():
    """Health check endpoint.

    Returns:
        dict: A status message indicating the service is operational.
    """
    return {"Pong"}


@app.exception_handler(PyJWTError)
def authjwt_exception_handler(request, exc):
    # Map “signature expired” to 401 so the UI knows to refresh/relogin
    status = 401 if "expired" in str(exc.message).lower() else exc.status_code
    return JSONResponse(status_code=status, content={"detail": exc.message})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
