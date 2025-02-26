from fastapi import FastAPI
from src.routes import progress
import uvicorn

app = FastAPI()
app.include_router(progress.router)

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
