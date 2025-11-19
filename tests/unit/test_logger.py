"""
Unit tests for the logging utilities.

Tests the logger setup, formatting, and configuration handling.
"""

import json
import logging

from src.utils.logger import ColoredFormatter, JsonFormatter, setup_logger


def test_setup_logger_default_config(tmp_path):
    """Test logger setup with default configuration."""
    log_file = tmp_path / "test.log"

    config = {
        "level": "INFO",
        "format": "json",
        "file": {
            "enabled": True,
            "path": str(log_file),
            "maxbytes": 1024,
            "backupcount": 3,
        },
        "console": {"enabled": False},
    }

    logger = setup_logger("test_logger", config)

    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0


def test_logger_writes_to_file(tmp_path):
    """Test that logger actually writes messages to file."""
    log_file = tmp_path / "test.log"

    config = {
        "level": "DEBUG",
        "format": "json",
        "file": {
            "enabled": True,
            "path": str(log_file),
            "maxbytes": 1024,
            "backupcount": 3,
        },
        "console": {"enabled": False},
    }

    logger = setup_logger("test_file_logger", config)
    logger.info("Test message")

    # Force flush
    for handler in logger.handlers:
        handler.flush()

    assert log_file.exists()
    content = log_file.read_text()
    assert len(content) > 0


def test_json_formatter():
    """Test JSON log formatter."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    log_data = json.loads(formatted)

    assert log_data["level"] == "INFO"
    assert log_data["message"] == "Test message"
    assert log_data["line"] == 42
    assert "timestamp" in log_data


def test_colored_formatter():
    """Test colored log formatter."""
    formatter = ColoredFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=42,
        msg="Error message",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)

    # Should contain ANSI color codes
    assert "\033[" in formatted
    assert "Error message" in formatted


def test_logger_levels(tmp_path):
    """Test that logger respects configured log levels."""
    log_file = tmp_path / "test.log"

    config = {
        "level": "WARNING",
        "format": "text",
        "file": {
            "enabled": True,
            "path": str(log_file),
            "maxbytes": 1024,
            "backupcount": 3,
        },
        "console": {"enabled": False},
    }

    logger = setup_logger("test_level_logger", config)

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")

    # Force flush
    for handler in logger.handlers:
        handler.flush()

    content = log_file.read_text()

    # Only WARNING and above should be logged
    assert "Debug message" not in content
    assert "Info message" not in content
    assert "Warning message" in content


def test_multiple_loggers_isolated(tmp_path):
    """Test that multiple loggers don't interfere with each other."""
    log_file1 = tmp_path / "logger1.log"
    log_file2 = tmp_path / "logger2.log"

    config1 = {
        "level": "INFO",
        "format": "text",
        "file": {
            "enabled": True,
            "path": str(log_file1),
            "maxbytes": 1024,
            "backupcount": 3,
        },
        "console": {"enabled": False},
    }

    config2 = {
        "level": "ERROR",
        "format": "text",
        "file": {
            "enabled": True,
            "path": str(log_file2),
            "maxbytes": 1024,
            "backupcount": 3,
        },
        "console": {"enabled": False},
    }

    logger1 = setup_logger("logger1", config1)
    logger2 = setup_logger("logger2", config2)

    logger1.info("Logger 1 info")
    logger2.error("Logger 2 error")

    # Force flush
    for handler in logger1.handlers + logger2.handlers:
        handler.flush()

    assert log_file1.exists()
    assert log_file2.exists()

    content1 = log_file1.read_text()
    content2 = log_file2.read_text()

    assert "Logger 1 info" in content1
    assert "Logger 2 error" in content2
    assert "Logger 2 error" not in content1
    assert "Logger 1 info" not in content2
