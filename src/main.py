import os
import sys

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import src.pipeline as pipeline
from src.command import Command, command_from_json, command_from_json_transcribe_version
from src.pipeline import assemble_prompt
from src.routes import agents, debug, progress, upload
from src.transcribe import transcribe_audio, transcribe_from_upload


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

config_api_url = os.getenv("RAGDOLL_CONFIG_API_URL", "http://localhost:3000")

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
        config_api_url
    ],  # Frontend URL(s), static rn for testing, but need env variable later
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


@app.get("/")
def hello_world():
    """Hello World.

    Returns:
        message: Hello World from FastAPI!
    """
    return {"message": "Hello World!"}


@app.get("/ping")
def ping():
    """Ping.

    Returns:
        status: Pong
    """
    return {"status": "PONG!"}


@app.post("/ask")
async def ask(request: Request):
    """Ask.

    Accepts a byte array input and returns a response.

    Returns:
        response: str
    """
    try:
        body_bytes = await request.body()

        data = body_bytes.decode("utf-8")

        command: Command = command_from_json(data)

        if command is None:
            return JSONResponse(
                content={"message": "Invalid command format."}, status_code=400
            )

        response = assemble_prompt(command)
        return response

    except UnicodeDecodeError:
        return JSONResponse(
            content={"message": "Invalid encoding. Expected UTF-8 encoded JSON."},
            status_code=400,
        )
    except Exception as e:
        return JSONResponse(
            content={"message": f"Error processing request: {e!s}"}, status_code=500
        )


@app.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(...),  # noqa: B008
    language: str = Form(None),
):
    """Transcribe an audio file to text.

    Parameters:
    - audio: The audio file (WAV format recommended)
    - language: Optional language code (e.g., 'en', 'fr', 'es')

    Returns:
    - A JSON response with transcription or error message
    """
    result = transcribe_audio(audio, language)

    if result["success"]:
        return JSONResponse(content=result, status_code=200)
    else:
        return JSONResponse(content=result, status_code=400)


@app.post("/askTranscribe")
async def ask_transcribe(
    audio: UploadFile = File(...),  # noqa: B008
    data: str = Form(...),
):
    """Transcribes an audio file and processes a command."""
    transcribed = transcribe_from_upload(audio)

    command: Command = command_from_json_transcribe_version(data, question=transcribed)
    if command is None:
        return {"message": "Invalid command."}

    response = assemble_prompt(command)
    return {"response": response}


@app.get("/getAnswerFromUser")
def get_answer_from_user(
    answer: str,
    target: str,
    question: str,
) -> str:
    """Get the answer from the user. Target is what the question is about. Example: "What is your name?" -> target= "name".

    Returns:
        response: str
    """
    response: str = pipeline.get_answer_from_user(answer, target, question)
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
