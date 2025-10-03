import os

from dotenv import load_dotenv


load_dotenv()


class Config:
    """Configuration class to manage environment variables and model loading."""

    def __init__(self, path=".env"):
        self.path = path
        self.ENV = os.getenv("ENV", "dev")

        self.MODEL = os.getenv("MODEL", "idun")
        self.GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        self.API_KEY = os.getenv("OPENAI_API_KEY", "your_default_api_key")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_default_gemini_api_key")
        self.IDUN_API_URL = os.getenv(
            "IDUN_API_URL", "https://idun-llm.hpc.ntnu.no/api/chat/completions"
        )
        self.IDUN_API_KEY = os.getenv("IDUN_API_KEY", "secret_secret")
        self.IDUN_MODEL = os.getenv("IDUN_MODEL", "openai/gpt-oss-120b")

        if self.ENV == "dev":  # TODO: change this to 'dev' when ready
            self.MONGODB_URI = os.getenv(
                "MOCK_MONGODB_URI", "mongodb://localhost:27017"
            )
            self.MONGODB_COLLECTION = os.getenv(
                "MOCK_MONGODB_COLLECTION", "test_collection"
            )
            self.MONGODB_DATABASE = os.getenv("MOCK_MONGODB_DATABASE", "test_database")
            self.RAG_DATABASE_SYSTEM = os.getenv("MOCK_RAG_DATABASE_SYSTEM", "mongodb")
        else:
            self.MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            self.MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "test_collection")
            self.MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "test_database")
            self.RAG_DATABASE_SYSTEM = os.getenv("RAG_DATABASE_SYSTEM", "mongodb")
