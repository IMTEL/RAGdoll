import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import Config
from src.routes import agents, chat, debug, progress, upload


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

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

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
# Create static directory if it doesn't exist (optional, good practice)
if not os.path.exists(static_dir):
    try:
        os.makedirs(static_dir)
        print(f"Created static directory at: {static_dir}")
    except OSError as e:
        print(f"Error creating static directory {static_dir}: {e}")

# Check if directory exists before mounting
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(
        f"Warning: Static directory not found at {static_dir}. Static files will not be served."
    )


# Route to serve the main HTML page
@app.get("/test-ui", response_class=HTMLResponse)
async def read_test_ui():
    html_file_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_file_path):
        with open(html_file_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    error_message = f"<h1>Error: Test UI HTML file not found.</h1><p>Expected location: {html_file_path}</p>"
    if not os.path.isdir(static_dir):
        error_message += (
            "<p>The static directory itself was not found or is not a directory.</p>"
        )
    return HTMLResponse(content=error_message, status_code=404)


# Progress router
app.include_router(progress.router)
# Failure router
# app.include_router(failure.router)
# Debug router
app.include_router(debug.router)
# RAG router
# app.include_router(rag.router)
# Upload router
app.include_router(upload.router)
# Agents router
app.include_router(agents.router)
# Chat router (handles /ask, /transcribe, /askTranscribe)
app.include_router(chat.router)


@app.get("/")
def hello_world():
    """Hello World.

    Returns:
        message: Hello World from FastAPI!
    """
    return {"message": "Hello World!"}


@app.get("/ping")
def ping():
    """Health check endpoint.

    Returns:
        status: Pong
    """
    return {"status": "PONG!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
