from fastapi import FastAPI
from src.command import Command, command_from_json
from src.pipeline import assemble_prompt
from src.routes import progress, failure, debug, rag, upload
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

app = FastAPI(
    title="Chat-Service Microservice API",
    description="Generate prompts with context and passing them to LLM.",
    version="1.0.0",

)
# Progress router
app.include_router(progress.router)
# Failure router
app.include_router(failure.router)
# Debug router
app.include_router(debug.router)
# RAG router
app.include_router(rag.router)
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
