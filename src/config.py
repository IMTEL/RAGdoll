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
        # Log user messages and responses (secrets masked). Off by default.
        self.LOG_CONVERSATIONS = (
            os.getenv("LOG_CONVERSATIONS", "false").strip().lower() == "true"
        )
        # Serve the speech and translation routes (transcribe, tts, stt/tts
        # warmup, translate*, transliterate*) without credentials, for XR/Unity
        # clients that do not send any. Off by default: those routes need a
        # user token or an agent access key.
        self.PUBLIC_SPEECH_ENDPOINTS = (
            os.getenv("PUBLIC_SPEECH_ENDPOINTS", "false").strip().lower() == "true"
        )
        self.RAGDOLL_CONFIG_API_URL = os.getenv(
            "RAGDOLL_CONFIG_API_URL", "http://localhost:3000"
        )
        self.RAGDOLL_CHAT_API_URL = os.getenv(
            "RAGDOLL_CHAT_API_URL", "http://localhost:3001"
        )
        self.CORS_ALLOWED_ORIGINS = [
            origin.strip().rstrip("/")
            for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
            if origin.strip()
        ]
        self.PROXY_PATH = os.getenv("PROXY_PATH", "/").strip("/")
        self.MODEL = os.getenv("MODEL", "idun")
        self.GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        self.IDUN_API_BASE_URL = os.getenv(
            "IDUN_API_BASE_URL", "https://llm.hpc.ntnu.no/v1"
        ).rstrip("/")
        self.IDUN_API_URL = os.getenv(
            "IDUN_API_URL", f"{self.IDUN_API_BASE_URL}/chat/completions"
        )
        if "idun-llm.hpc.ntnu.no" in self.IDUN_API_URL:
            self.IDUN_API_URL = f"{self.IDUN_API_BASE_URL}/chat/completions"
        self.IDUN_API_KEY = os.getenv("IDUN_API_KEY", "secret_secret")
        self.ACCESS_SERVICE = os.getenv("ACCESS_SERVICE", "service")
        self.IDUN_MODEL = os.getenv("IDUN_MODEL", "openai/gpt-oss-120b")
        # Seconds before an LLM or embedding HTTP call gives up.
        self.LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        # Whisper transcriptions / Piper syntheses allowed to run at once.
        self.STT_MAX_CONCURRENCY = int(os.getenv("STT_MAX_CONCURRENCY", "2"))
        self.TTS_MAX_CONCURRENCY = int(os.getenv("TTS_MAX_CONCURRENCY", "2"))

        self.TTS_ENGINE = os.getenv("TTS_ENGINE", "piper").lower()
        self.TTS_VOICE_DIR = os.getenv("TTS_VOICE_DIR", "/data/piper/voices")
        self.TTS_DEFAULT_LANGUAGE = os.getenv("TTS_DEFAULT_LANGUAGE", "en").lower()
        self.TTS_DEFAULT_VOICE_EN = os.getenv(
            "TTS_DEFAULT_VOICE_EN", "en_US-lessac-medium"
        )
        self.TTS_DEFAULT_VOICE_NO = os.getenv(
            "TTS_DEFAULT_VOICE_NO", "no_NO-talesyntese-medium"
        )
        self.TTS_DEFAULT_VOICE_ES = os.getenv(
            "TTS_DEFAULT_VOICE_ES", "es_ES-davefx-medium"
        )
        self.TTS_DEFAULT_VOICE_ZH = os.getenv(
            "TTS_DEFAULT_VOICE_ZH", "zh_CN-huayan-medium"
        )
        self.TTS_USE_CUDA = os.getenv("TTS_USE_CUDA", "false").lower() == "true"
        self.TTS_WARMUP_TEXT = os.getenv("TTS_WARMUP_TEXT", "Ready.")
        self.TRANSLATION_PROVIDER = os.getenv(
            "TRANSLATION_PROVIDER", "libretranslate"
        ).lower()
        self.TRANSLATION_BASE_URL = os.getenv(
            "TRANSLATION_BASE_URL", "http://libretranslate:5000"
        )
        self.TRANSLATION_API_KEY = os.getenv("TRANSLATION_API_KEY", "")
        self.TRANSLATION_TIMEOUT_SECONDS = float(
            os.getenv("TRANSLATION_TIMEOUT_SECONDS", "30")
        )
        self.TRANSLATION_CACHE_TTL_SECONDS = int(
            os.getenv("TRANSLATION_CACHE_TTL_SECONDS", "3600")
        )
        self.TRANSLATION_CACHE_MAX_ENTRIES = int(
            os.getenv("TRANSLATION_CACHE_MAX_ENTRIES", "2000")
        )

        self.MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

        self.RAG_DATABASE_SYSTEM = os.getenv(
            self._prod_or_mock_env("RAG_DATABASE_SYSTEM"), "mongodb"
        )

        self.MONGODB_URI = os.getenv(
            self._prod_or_mock_env("MONGODB_URI"), "mongodb://localhost:27017"
        )
        if self.RUNNING_TESTS:
            # Never let tests touch the database named by the production variable.
            self.MONGODB_DATABASE = os.getenv("MOCK_MONGODB_DATABASE", "test_database")
        else:
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
        self.MONGODB_USER_COLLECTION = os.getenv("MONGODB_USER_COLLECTION", "users")
        self.MONGODB_PROGRESS_COLLECTION = os.getenv(
            "MONGODB_PROGRESS_COLLECTION", "progress"
        )
        self.MONGODB_PROFILES_COLLECTION = os.getenv(
            "MONGODB_PROFILES_COLLECTION", "job_seeker_profiles"
        )
        self.MONGODB_FEEDBACK_COLLECTION = os.getenv(
            "MONGODB_FEEDBACK_COLLECTION", "message_feedback"
        )
        self.MONGODB_TEMPLATE_COLLECTION = os.getenv(
            "MONGODB_TEMPLATE_COLLECTION", "templates"
        )
        self.MONGODB_UPLOAD_TASKS_COLLECTION = os.getenv(
            "MONGODB_UPLOAD_TASKS_COLLECTION", "upload_tasks"
        )
        self.PROGRESS_TTL_HOURS = float(os.getenv("PROGRESS_TTL_HOURS", "24"))

        self.AUTH_SERVICE = os.getenv("AUTH_SERVICE", "service")

        self.KEYCLOAK_BASE_URL = os.getenv(
            "KEYCLOAK_BASE_URL", "http://localhost:8080"
        ).rstrip("/")
        self.KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "ragdoll")
        self.KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "ragdoll-config")
        self.KEYCLOAK_ISSUER = os.getenv(
            "KEYCLOAK_ISSUER",
            f"{self.KEYCLOAK_BASE_URL}/realms/{self.KEYCLOAK_REALM}",
        ).rstrip("/")
        self.KEYCLOAK_ALLOWED_ISSUERS = [
            issuer.strip().rstrip("/")
            for issuer in os.getenv("KEYCLOAK_ALLOWED_ISSUERS", "").split(",")
            if issuer.strip()
        ]

        self.GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "set your id here")
        self.GOOGLE_CLIENT_SECRET = os.getenv(
            "GOOGLE_CLIENT_SECRET", "set your secret here"
        )

        # Shared key The Candidate uses to list and chat with published agents.
        # Empty disables it.
        self.CANDIDATE_CLIENT_KEY = os.getenv("CANDIDATE_CLIENT_KEY", "").strip()

        # The observer agent (feedback after an interview) The Candidate may also
        # chat with using its client key. It is not published, so it is never
        # listed as an interviewer. Empty gives access to no observer.
        self.CANDIDATE_OBSERVER_AGENT_ID = os.getenv(
            "CANDIDATE_OBSERVER_AGENT_ID", ""
        ).strip()

        self.FERNET_KEY = os.getenv("FERNET_KEY", "").strip()

        if not self.FERNET_KEY:
            raise RuntimeError("FERNET_KEY is missing from environment variables")

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
            self.MONGODB_PROFILES_COLLECTION,
            self.MONGODB_FEEDBACK_COLLECTION,
            # TODO: add document collection
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
