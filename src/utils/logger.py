"""
Logging utilities for the anomaly detection system.

Provides centralized logging configuration with support for
JSON and text formats, file rotation, and configurable log levels.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON string.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=None).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Format log records with ANSI color codes for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with color codes.

        Args:
            record: The log record to format

        Returns:
            Color-formatted log string
        """
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    config: Optional[dict[str, Any]] = None,
) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name: Name for the logger (typically __name__ of the module)
        config: Optional logging configuration dictionary.
                If None, uses default configuration.

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing started")
    """
    # Default configuration
    default_config = {
        "level": "INFO",
        "format": "json",
        "file": {
            "enabled": True,
            "path": "logs/anomaly_detection.log",
            "maxbytes": 10485760,  # 10MB
            "backupcount": 10,
        },
        "console": {
            "enabled": True,
            "colorize": True,
        },
    }

    # Merge with provided config
    if config:
        default_config.update(config)
    config = default_config

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config["level"].upper()))
    logger.handlers.clear()  # Remove existing handlers

    # File handler with rotation
    if config["file"]["enabled"]:
        log_path = Path(config["file"]["path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=config["file"]["maxbytes"],
            backupCount=config["file"]["backupcount"],
        )

        if config["format"] == "json":
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

        logger.addHandler(file_handler)

    # Console handler
    if config["console"]["enabled"]:
        console_handler = logging.StreamHandler(sys.stdout)

        if config["console"].get("colorize") and config["format"] != "json":
            console_handler.setFormatter(
                ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
        elif config["format"] == "json":
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default config.

    Args:
        name: Name of the logger

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
