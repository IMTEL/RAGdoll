import os


def build_logging_config() -> dict:
    """Build the logging config from LOG_LEVEL and LOG_FILE.

    Logs go to stdout only. A rotating file (10 MB x 5) is written only when
    LOG_FILE is set. Chatty client libraries are held at WARNING so they never
    log request payloads or credentials.
    """
    level = os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    log_file = os.getenv("LOG_FILE", "").strip()

    handlers: dict = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": level,
        },
    }
    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
            "level": level,
        }

    quiet = ["pymongo", "httpx", "httpcore", "openai", "urllib3", "google", "faiss"]
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
            },
        },
        "handlers": handlers,
        "loggers": {name: {"level": "WARNING"} for name in quiet},
        "root": {"handlers": list(handlers), "level": level},
    }


LOGGING_CONFIG = build_logging_config()

SUPPORTED_LLM_PROVIDERS = [
    {"id": "idun", "label": "Idun"},
    {"id": "openai", "label": "OpenAI"},
    {"id": "gemini", "label": "Google Gemini"},
]

SUPPORTED_EMBEDDING_PROVIDERS = [
    {"id": "idun", "label": "Idun"},
    {"id": "openai", "label": "OpenAI"},
    {"id": "gemini", "label": "Google Gemini"},
]
