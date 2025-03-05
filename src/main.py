from fastapi import FastAPI
from src.routes import progress, failure, debug
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

if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8000)
