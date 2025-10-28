import os
from typing import ClassVar

from dotenv import load_dotenv


# Initial load of environment variables from .env file
load_dotenv()


class Config:
    """Configuration class to manage environment variables and model loading."""

    _instances: ClassVar[dict[type, "Config"]] = {}
    _initialized: ClassVar[bool] = False  # Set to True after first initialization

    def __new__(cls):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]

    def __init__(self):
        """Load environment variables and set configuration attributes."""
        if Config._initialized:
            return  # Avoid re-initialization

        Config._initialized = True

        self.ENV = os.getenv("ENV", "dev")

        # Flag to indicate if tests are running.
        # Environment variable is set in pyproject.toml under [tool.pytest_env]
        self.RUNNING_TESTS = os.getenv("RUNNING_TESTS", "false").lower() == "true"

        if self.RUNNING_TESTS:  # Ensure tests always run in 'dev' environment
            self.ENV = "dev"
        self.RAGDOLL_CONFIG_API_URL = os.getenv(
            "RAGDOLL_CONFIG_API_URL", "http://localhost:3000"
        )
        self.RAGDOLL_CHAT_API_URL = os.getenv(
            "RAGDOLL_CHAT_API_URL", "http://localhost:3001"
        )

        self.MODEL = os.getenv("MODEL", "idun")
        self.GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        self.API_KEY = os.getenv("OPENAI_API_KEY", "your_default_api_key")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_default_gemini_api_key")
        self.IDUN_API_URL = os.getenv(
            "IDUN_API_URL", "https://idun-llm.hpc.ntnu.no/api/chat/completions"
        )
        self.IDUN_API_KEY = os.getenv("IDUN_API_KEY", "secret_secret")
        self.ACCESS_SERVICE = os.getenv("ACCESS_SERVICE", "service")
        self.IDUN_MODEL = os.getenv("IDUN_MODEL", "openai/gpt-oss-120b")

        self.RAG_DATABASE_SYSTEM = os.getenv(
            self._prod_or_mock_env("RAG_DATABASE_SYSTEM"), "mongodb"
        )

        self.MONGODB_URI = os.getenv(
            self._prod_or_mock_env("MONGODB_URI"), "mongodb://localhost:27017"
        )
        self.MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "test_database")

        # It is expected now that agents and contexts are in the same database
        # and in separate collections
        self.MONGODB_DOCUMENTS_COLLECTION = os.getenv(
            "MONGODB_DOCUMENTS_COLLECTION", "documents"
        )
        self.MONGODB_CONTEXT_COLLECTION = os.getenv(
            "MONGODB_CONTEXT_COLLECTION", "contexts"
        )
        self.MONGODB_AGENT_COLLECTION = os.getenv("MONGODB_AGENT_COLLECTION", "agents")

        # TODO: Fetch models from IDUN's endpoint
        self.IDUN_MODELS = os.getenv(
            "IDUN_MODELS", "Qwen3-Coder-30B-A3B-Instruct,openai/gpt-oss-120b"
        ).split(",")

        self.FERNET_KEY = os.getenv("FERNET_KEY", "")

        if not self.FERNET_KEY:
            raise RuntimeError("FERNET_KEY is missing from environment variables")

        self.FERNET_KEY = self.FERNET_KEY.strip()

        self._validate_collection_names()

    def _prod_or_mock_env(self, env_var: str) -> str:
        """Determine whether to use production or mock environment variable.

        Args:
            env_var (str): The environment variable to check.

        Returns:
            str: "MOCK_" + env_var if in dev mode, else env_var.
        """
        if self.ENV == "dev":
            return "MOCK_" + env_var

        return env_var

    def _validate_collection_names(self):
        """Validate that MongoDB collection names are mutually exclusive."""
        names = [
            self.MONGODB_CONTEXT_COLLECTION,
            self.MONGODB_AGENT_COLLECTION,
            self.MONGODB_DOCUMENTS_COLLECTION,
        ]

        total_names = len(names)

        if len(set(names)) != total_names:
            raise ValueError(
                "MongoDB collection names must be mutually exclusive. "
                f"Current names: {names}"
            )

    @staticmethod
    def _delete_instance__():
        """Delete the singleton instance for testing purposes."""
        Config._instances = {}
        Config._initialized = False
