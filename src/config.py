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

        self.MONGODB_URI = os.getenv(
            get_mock_or_real_env("MONGODB_URI"), "mongodb://localhost:27017"
        )
        self.MONGODB_DATABASE = os.getenv(
            get_mock_or_real_env("MONGODB_DATABASE"), "test_database"
        )
        self.MONGODB_CONTEXT_COLLECTION = os.getenv(
            get_mock_or_real_env("MONGODB_CONTEXT_COLLECTION"), "test_collection"
        )
        # It is expected now that agents and contexts are in separate collections
        # in the same database
        self.MONGODB_AGENTS_COLLECTION = os.getenv(
            get_mock_or_real_env("MONGODB_AGENTS_COLLECTION"), "agents"
        )
        self.RAG_DATABASE_SYSTEM = os.getenv(
            get_mock_or_real_env("RAG_DATABASE_SYSTEM"), "mongodb"
        )


def get_mock_or_real_env(env_var: str) -> str:
    """Determine whether to use mock or real database based on environment variable.

    Args:
        env_var (str): The environment variable to check.

    Returns:
        str: "MOCK_" + env_var if in dev mode, else env_var.
    """
    env = os.getenv("ENV", "dev")
    if env == "dev":
        return "MOCK_" + env_var

    return env_var
