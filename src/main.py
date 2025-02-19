from fastapi import FastAPI
import uvicorn

app = FastAPI()

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
