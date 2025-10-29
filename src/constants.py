LOGGING_CONFIG = {
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
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": "app.log",
            "level": "DEBUG",
        },
    },
    "loggers": {
        # Set specific loggers to INFO level for cleaner output
        "src.scraper_service.scraper": {
            "level": "INFO",
        },
        "src.pipeline": {
            "level": "INFO",
        },
        "src.rag_service.dao.context.mongodb_context_dao": {
            "level": "INFO",
        },
        # Silence noisy third-party libraries
        "pymongo": {
            "level": "WARNING",
        },
        "urllib3": {
            "level": "WARNING",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}
