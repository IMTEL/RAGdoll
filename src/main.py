from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import sys
import os
from tempfile import NamedTemporaryFile

from src.command import Command, command_from_json
from src.pipeline import assemble_prompt
from src.routes import progress, debug, upload
from src.transcribe import transcribe_from_upload
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

app = FastAPI(
    title="Chat-Service Microservice API",
    description="Generate prompts with context and passing them to LLM.",
    version="1.0.0",

)
# Progress router
app.include_router(progress.router)
# Debug router
app.include_router(debug.router)
# Upload router
app.include_router(upload.router)

@app.get("/")
def hello_world():
    """Hello World

    Returns:
        message: Hello World from FastAPI!
    """
    return {"message": "Hello World!"}


@app.get("/ping")
def ping():
    """Ping

    Returns:
        status: Pong
    """
    return {"status": "PONG!"}


@app.get("/ask")
def ask(
    data: str
) -> str:
    """Ask

    Returns:
        response: str
    """
    command: Command = command_from_json(data)
    if command is None:
        return {"message": "Invalid command."}
    response = assemble_prompt(command)
    return {"response": response}

@app.post("/askTranscribe")
async def askTranscribe(
    audio: UploadFile = File(...),
    data: str = Form(...)
):
    """
    Transcribes an audio file and processes a command.
    """
    transcribed = transcribe_from_upload(audio)

   
    command: Command = command_from_json(data, question=transcribed)
    if command is None:
        return {"message": "Invalid command."}

    response = assemble_prompt(command)
    return {"response": response}

@app.get("/getAnswerFromUser")
def getAnswerFromUser(
    answer: str,
    target: str,
    question: str,
) -> str:
    """Get the answer from the user. Target is what the question is about. Example: "What is your name?" -> target= "name".

    Returns:
        response: str
    """
    response: str = getAnswerFromUser(answer, target, question)
    return {"response": response}

if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8000)
