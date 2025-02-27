from fastapi import FastAPI
from src.routes import progress, failure, debug
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


app = FastAPI()
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
    return {"message": "Hello World from FastAPI!"}

@app.get("/ping")
def ping():
    """Ping

    Returns:
        status: I AM ALIVE!
    """
    return {"status": "I AM ALIVE!"}

if __name__ == "__main__":  # This should always be last
    uvicorn.run(app, host="0.0.0.0", port=8000)
