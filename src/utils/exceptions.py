"""Custom exceptions for the anomaly detection system.

This module defines the exception hierarchy for data loading and validation errors.
"""


class AnomalyDetectionError(Exception):
    """Base exception for all anomaly detection system errors."""

    pass


class DataLoadError(AnomalyDetectionError):
    """Raised when data loading fails.

    This includes failures from SQL queries, file I/O, or connection issues.
    """

    pass


class DataValidationError(AnomalyDetectionError):
    """Raised when validation fails critically.

    This is used for validation failures that prevent further processing.
    """

    pass


class ConfigurationError(AnomalyDetectionError):
    """Raised when configuration is invalid or missing.

    This includes missing required configuration keys or invalid config values.
    """

    pass
