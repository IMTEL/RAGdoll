import logging
from logging.config import dictConfig

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import Config
from src.constants import LOGGING_CONFIG
from src.routes import agents, auth, chat, debug, progress, upload


# Apply the logging configuration
dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Chat-Service Microservice API",
    description="Generate prompts with context and passing them to LLM.",
    version="1.0.0",
)

# CORS middleware
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        Config().RAGDOLL_CONFIG_API_URL,
        Config().RAGDOLL_CHAT_API_URL,
    ],  # TODO: Frontend URL(s), static rn for testing, but need env variable later
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

# Chat router (handles /ask, /transcribe, /askTranscribe)
app.include_router(chat.router)

# Authentication router
app.include_router(auth.router)


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
    return {"status": "Service is operational."}


from fastapi_jwt_auth.exceptions import AuthJWTException


@app.exception_handler(AuthJWTException)
def authjwt_exception_handler(request, exc):
    # Map “signature expired” to 401 so the UI knows to refresh/relogin
    status = 401 if "expired" in str(exc.message).lower() else exc.status_code
    return JSONResponse(status_code=status, content={"detail": exc.message})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
