"""
Unit tests for configuration loader.
"""

import os
from tempfile import NamedTemporaryFile

import pytest

from src.utils.config_loader import load_config, substitute_env_vars, validate_config
from src.utils.exceptions import ConfigurationError


def test_substitute_env_vars_simple():
    """Test simple environment variable substitution."""
    os.environ["TEST_VAR"] = "test_value"
    config = {"key": "${TEST_VAR}"}

    result = substitute_env_vars(config)

    assert result["key"] == "test_value"
    del os.environ["TEST_VAR"]


def test_substitute_env_vars_nested():
    """Test nested environment variable substitution."""
    os.environ["DB_HOST"] = "localhost"
    os.environ["DB_PORT"] = "5432"
    config = {
        "database": {
            "host": "${DB_HOST}",
            "port": "${DB_PORT}",
            "connection": "host=${DB_HOST}:${DB_PORT}",
        }
    }

    result = substitute_env_vars(config)

    assert result["database"]["host"] == "localhost"
    assert result["database"]["port"] == "5432"
    assert result["database"]["connection"] == "host=localhost:5432"

    del os.environ["DB_HOST"]
    del os.environ["DB_PORT"]


def test_substitute_env_vars_in_list():
    """Test environment variable substitution in lists."""
    os.environ["LIST_VAR"] = "value"
    config = {"items": ["${LIST_VAR}", "static", "${LIST_VAR}"]}

    result = substitute_env_vars(config)

    assert result["items"] == ["value", "static", "value"]
    del os.environ["LIST_VAR"]


def test_substitute_env_vars_missing_raises_error():
    """Test that missing environment variable raises ConfigurationError."""
    config = {"key": "${MISSING_VAR}"}

    with pytest.raises(
        ConfigurationError, match="Environment variable 'MISSING_VAR' is not set"
    ):
        substitute_env_vars(config)


def test_substitute_env_vars_preserves_non_strings():
    """Test that non-string values are preserved."""
    config = {"number": 42, "boolean": True, "null": None, "float": 3.14}

    result = substitute_env_vars(config)

    assert result == config


def test_validate_config_model_config():
    """Test validation of model configuration."""
    # Valid model config
    valid_config = {"features": {}, "isolationforest": {}}
    validate_config(valid_config)  # Should not raise

    # Missing isolationforest section
    invalid_config = {"features": {}}
    with pytest.raises(ConfigurationError, match="Missing required model config"):
        validate_config(invalid_config)

    # Missing features section
    invalid_config = {"isolationforest": {}}
    with pytest.raises(ConfigurationError, match="Missing required model config"):
        validate_config(invalid_config)


def test_validate_config_data_config():
    """Test validation of data configuration."""
    # Valid data config
    valid_config = {"datasource": {}, "schema": {}, "validation": {}}
    validate_config(valid_config)  # Should not raise

    # Missing sections
    invalid_config = {"datasource": {}}
    with pytest.raises(ConfigurationError, match="Missing required data config"):
        validate_config(invalid_config)


def test_validate_config_empty():
    """Test that empty config passes validation."""
    empty_config: dict = {}
    validate_config(empty_config)  # Should not raise


def test_load_config_success():
    """Test successful configuration loading and merging."""
    # Create temporary config files
    model_config = """
features:
  top_mcc_count: 10
isolationforest:
  nestimators: 100
logging:
  level: INFO
"""

    data_config = """
datasource:
  type: sql
schema:
  required_columns: []
validation:
  rules: {}
"""

    with (
        NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as model_file,
        NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as data_file,
    ):

        model_file.write(model_config)
        model_file.flush()

        data_file.write(data_config)
        data_file.flush()

        try:
            config = load_config(model_file.name, data_file.name)

            # Check merged config contains keys from both files
            assert "features" in config
            assert "isolationforest" in config
            assert "logging" in config
            assert "datasource" in config
            assert "schema" in config
            assert "validation" in config

        finally:
            os.unlink(model_file.name)
            os.unlink(data_file.name)


def test_load_config_with_env_substitution():
    """Test configuration loading with environment variable substitution."""
    os.environ["DB_CONNECTION"] = "postgresql://localhost:5432/testdb"

    model_config = """
features:
  top_mcc_count: 10
isolationforest:
  nestimators: 100
"""

    data_config = """
datasource:
  type: sql
  connection: ${DB_CONNECTION}
schema:
  required_columns: []
validation:
  rules: {}
"""

    with (
        NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as model_file,
        NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as data_file,
    ):

        model_file.write(model_config)
        model_file.flush()

        data_file.write(data_config)
        data_file.flush()

        try:
            config = load_config(model_file.name, data_file.name)

            assert (
                config["datasource"]["connection"]
                == "postgresql://localhost:5432/testdb"
            )

        finally:
            os.unlink(model_file.name)
            os.unlink(data_file.name)
            del os.environ["DB_CONNECTION"]


def test_load_config_file_not_found():
    """Test that FileNotFoundError is raised for missing config files."""
    with pytest.raises(FileNotFoundError, match="Model config file not found"):
        load_config("nonexistent_model.yaml", "config/dataconfig.yaml")

    with pytest.raises(FileNotFoundError, match="Data config file not found"):
        load_config("config/modelconfig.yaml", "nonexistent_data.yaml")


def test_load_config_invalid_yaml():
    """Test that ConfigurationError is raised for invalid YAML."""
    invalid_yaml = """
invalid: yaml: content:
  - broken
  nested:
"""

    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(invalid_yaml)
        f.flush()

        try:
            with pytest.raises(ConfigurationError, match="Failed to parse"):
                load_config(f.name, "config/dataconfig.yaml")
        finally:
            os.unlink(f.name)


def test_load_config_not_dict():
    """Test that ConfigurationError is raised if config is not a dictionary."""
    list_config = "- item1\n- item2\n"

    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(list_config)
        f.flush()

        try:
            with pytest.raises(ConfigurationError, match="must be a YAML dictionary"):
                load_config(f.name, "config/dataconfig.yaml")
        finally:
            os.unlink(f.name)


def test_load_config_missing_env_var():
    """Test that ConfigurationError is raised for missing environment variables."""
    model_config = """
features:
  top_mcc_count: 10
isolationforest:
  nestimators: 100
"""

    data_config = """
datasource:
  type: sql
  connection: ${MISSING_DB_VAR}
schema:
  required_columns: []
validation:
  rules: {}
"""

    with (
        NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as model_file,
        NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as data_file,
    ):

        model_file.write(model_config)
        model_file.flush()

        data_file.write(data_config)
        data_file.flush()

        try:
            with pytest.raises(
                ConfigurationError, match="Environment variable 'MISSING_DB_VAR'"
            ):
                load_config(model_file.name, data_file.name)
        finally:
            os.unlink(model_file.name)
            os.unlink(data_file.name)
