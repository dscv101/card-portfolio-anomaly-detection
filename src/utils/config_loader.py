"""
Configuration loader for anomaly detection system.

This module provides utilities for loading and merging YAML configuration
files with environment variable substitution and validation.
"""

import os
import re
from pathlib import Path
from typing import Any, Union

import yaml

from src.utils.exceptions import ConfigurationError


def substitute_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively substitute environment variables in configuration values.

    Environment variables should be specified as ${VAR_NAME} in the config.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration dictionary with substituted values

    Raises:
        ConfigurationError: If a required environment variable is not set
    """
    if isinstance(config, dict):
        return {k: substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Find all ${VAR_NAME} patterns
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, config)

        result = config
        for var_name in matches:
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ConfigurationError(
                    f"Environment variable '{var_name}' is not set but required"
                )
            result = result.replace(f"${{{var_name}}}", env_value)
        return result
    else:
        return config


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate that required configuration sections are present.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ConfigurationError: If required sections are missing
    """
    # Check for model config sections
    if "isolationforest" in config or "features" in config:
        required_model_sections = ["features", "isolationforest"]
        missing = [s for s in required_model_sections if s not in config]
        if missing:
            raise ConfigurationError(
                f"Missing required model config sections: {', '.join(missing)}"
            )

    # Check for data config sections
    if "datasource" in config or "schema" in config:
        required_data_sections = ["datasource", "schema", "validation"]
        missing = [s for s in required_data_sections if s not in config]
        if missing:
            raise ConfigurationError(
                f"Missing required data config sections: {', '.join(missing)}"
            )


def load_config(
    model_config_path: Union[str, Path], data_config_path: Union[str, Path]
) -> dict[str, Any]:
    """
    Load and merge configuration files with environment variable substitution.

    Supports:
    - YAML file loading
    - Environment variable substitution: ${VAR_NAME}
    - Required section validation
    - Config merging

    Args:
        model_config_path: Path to model configuration YAML file
        data_config_path: Path to data configuration YAML file

    Returns:
        Merged configuration dictionary

    Raises:
        ConfigurationError: If config files cannot be loaded or are invalid
        FileNotFoundError: If config files do not exist
    """
    model_config_path = Path(model_config_path)
    data_config_path = Path(data_config_path)

    # Check files exist
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")
    if not data_config_path.exists():
        raise FileNotFoundError(f"Data config file not found: {data_config_path}")

    # Load YAML files
    try:
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse model config: {e}") from e

    try:
        with open(data_config_path) as f:
            data_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse data config: {e}") from e

    # Validate loaded configs are dicts
    if not isinstance(model_config, dict):
        raise ConfigurationError("Model config must be a YAML dictionary")
    if not isinstance(data_config, dict):
        raise ConfigurationError("Data config must be a YAML dictionary")

    # Substitute environment variables
    try:
        model_config = substitute_env_vars(model_config)
        data_config = substitute_env_vars(data_config)
    except ConfigurationError:
        raise

    # Merge configs (data config keys take precedence if there's overlap)
    merged_config: dict[str, Any] = {**model_config, **data_config}

    # Validate required sections
    validate_config(merged_config)

    return merged_config
