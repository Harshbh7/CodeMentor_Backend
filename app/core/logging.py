"""
CodeMentor AI - Structured Logging
=====================================
Configures structured logging using Python's standard `logging` module
with optional JSON formatting for production log aggregation (e.g., ELK, GCP Logging).

Design Rationale:
- JSON logs in production are machine-parseable by log aggregators.
- Human-readable text logs in development improve developer experience.
- Structured logs carry context (request_id, user_id) for tracing.
- We wrap Python's logging API so the rest of the app uses a consistent interface.
"""

import logging
import sys
from typing import Any

from app.core.config import get_settings

settings = get_settings()


class ColoredFormatter(logging.Formatter):
    """
    Terminal-friendly colored formatter for development.
    Adds ANSI color codes based on log level.
    """

    COLORS: dict[str, str] = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        record.levelname = f"{color}{record.levelname:<8}{reset}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """
    JSON structured log formatter for production environments.
    Each log line is a valid JSON object parseable by log aggregators.
    """

    def format(self, record: logging.LogRecord) -> str:
        import json
        import traceback

        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include exception info if present
        if record.exc_info:
            log_entry["exception"] = traceback.format_exception(*record.exc_info)

        # Include any extra fields added via logger.info("msg", extra={"key": "val"})
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


def setup_logging() -> None:
    """
    Configure the root logger and application-specific loggers.

    Call this once at application startup (in lifespan or main.py).
    After calling this, use `get_logger(__name__)` in any module.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Remove all existing handlers (avoids duplicate logs in hot-reload)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if settings.log_format == "json":
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    root_logger.info(
        "Logging configured: level=%s format=%s",
        settings.log_level,
        settings.log_format,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Factory to get a named logger.

    Usage:
        from app.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened", extra={"user_id": 42})

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(name)
