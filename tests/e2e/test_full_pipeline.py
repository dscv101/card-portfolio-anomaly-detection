"""
End-to-end tests for the full anomaly detection pipeline.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from main import run_anomaly_detection


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing."""
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C001", "C002", "C002", "C003"],
            "reporting_week": ["2025-11-18"] * 5,
            "mcc": ["5411", "5912", "5411", "5912", "5411"],
            "spend_amount": [100.0, 50.0, 200.0, 75.0, 150.0],
            "transaction_count": [5, 2, 10, 3, 7],
            "avg_ticket_amount": [20.0, 25.0, 20.0, 25.0, 21.43],
        }
    )


@pytest.fixture
def temp_config_files():
    """Create temporary configuration files for testing."""
    with TemporaryDirectory() as tmpdir:
        # Create model config
        model_config = """
features:
  top_mcc_count: 5
  lookback_windows: [4]
  concentration_metric: herfindahl
  handle_missing_history: impute_zero

isolationforest:
  nestimators: 10
  contamination: 0.3
  maxsamples: 256
  randomstate: 42
  maxfeatures: 1.0
  bootstrap: false

reporting:
  topnanomalies: 10
  rules:
    minspend: 50.0
    mintransactions: 1
  outputformat:
    type: csv
    includefeatures: true
    includescores: true

logging:
  level: INFO
  format: json
"""

        data_config = """
datasource:
  type: csv
  csv:
    directory: ./datasamples
    filename_pattern: sample_week_current.parquet

schema:
  required_columns:
    - customer_id: {type: string, nullable: false}
    - reporting_week: {type: date, nullable: false}
    - mcc: {type: string, nullable: false}
    - spend_amount: {type: float, nullable: false}
    - transaction_count: {type: int, nullable: false}
    - avg_ticket_amount: {type: float, nullable: false}

validation:
  rules:
    missing_critical: {severity: CRITICAL, action: reject_row}
    negative_spend: {severity: ERROR, action: flag_for_review}
    negative_txn_count: {severity: ERROR, action: flag_for_review}
    avg_ticket_mismatch: {severity: WARNING, action: recalculate, tolerance: 0.01}
    duplicates: {severity: ERROR, action: keep_first}
    extreme_ticket: {severity: WARNING, action: flag, multiplier: 10}
"""

        model_path = Path(tmpdir) / "modelconfig.yaml"
        data_path = Path(tmpdir) / "dataconfig.yaml"

        with open(model_path, "w") as f:
            f.write(model_config)

        with open(data_path, "w") as f:
            f.write(data_config)

        yield str(model_path), str(data_path)


def test_run_anomaly_detection_success(temp_config_files):
    """Test successful execution of the full pipeline."""
    model_config, data_config = temp_config_files

    # Run pipeline with existing sample data
    result = run_anomaly_detection(
        reporting_week="2025-11-18",
        config_path=model_config,
        data_config_path=data_config,
        mode="weekly",
    )

    # Verify execution summary structure
    assert "status" in result
    assert result["status"] == "success"

    # Verify timing information
    assert "reporting_week" in result
    assert result["reporting_week"] == "2025-11-18"
    assert "execution_time_seconds" in result
    assert result["execution_time_seconds"] >= 0

    # Verify data metrics
    assert "data_loaded_rows" in result
    assert result["data_loaded_rows"] > 0
    assert "valid_rows" in result
    assert "customers_scored" in result

    # Verify output files
    assert "output_files" in result
    assert "report" in result["output_files"]
    assert "summary" in result["output_files"]
    assert "model_artifacts" in result["output_files"]

    # Verify files exist
    assert Path(result["output_files"]["report"]).exists()
    assert Path(result["output_files"]["summary"]).exists()

    # Verify validation summary
    assert "validation_summary" in result

    # Verify config snapshot
    assert "config_snapshot" in result
    assert result["config_snapshot"]["mode"] == "weekly"


def test_run_anomaly_detection_missing_config():
    """Test pipeline failure with missing configuration files."""
    result = run_anomaly_detection(
        reporting_week="2025-11-18",
        config_path="nonexistent_model.yaml",
        data_config_path="nonexistent_data.yaml",
        mode="weekly",
    )

    assert result["status"] == "failed"
    assert "error" in result
    assert "error_type" in result
    # Could be configuration or file not found error
    assert result["error_type"] in ["configuration", "unexpected"]


def test_run_anomaly_detection_different_modes():
    """Test pipeline with different execution modes."""
    modes = ["weekly", "monthly", "adhoc"]

    for mode in modes:
        result = run_anomaly_detection(
            reporting_week="2025-11-18",
            config_path="config/modelconfig.yaml",
            data_config_path="config/dataconfig.yaml",
            mode=mode,
        )

        # All modes should work with valid data
        if result["status"] == "success":
            assert result["config_snapshot"]["mode"] == mode


def test_pipeline_output_file_structure():
    """Test that pipeline creates expected output file structure."""
    result = run_anomaly_detection(
        reporting_week="2025-11-18",
        config_path="config/modelconfig.yaml",
        data_config_path="config/dataconfig.yaml",
    )

    if result["status"] == "success":
        # Check report file
        report_path = Path(result["output_files"]["report"])
        assert report_path.suffix == ".csv"
        assert report_path.exists()
        assert report_path.stat().st_size > 0

        # Check summary file
        summary_path = Path(result["output_files"]["summary"])
        assert summary_path.suffix == ".json"
        assert summary_path.exists()
        assert summary_path.stat().st_size > 0

        # Check model artifacts directory
        artifacts_dir = Path(result["output_files"]["model_artifacts"])
        assert artifacts_dir.exists()
        assert artifacts_dir.is_dir()


def test_pipeline_logging(caplog):
    """Test that pipeline logs expected messages."""
    import logging

    caplog.set_level(logging.INFO)

    result = run_anomaly_detection(
        reporting_week="2025-11-18",
        config_path="config/modelconfig.yaml",
        data_config_path="config/dataconfig.yaml",
    )

    # Check for key log messages
    log_messages = [record.message for record in caplog.records]

    if result["status"] == "success":
        assert any("Starting anomaly detection pipeline" in msg for msg in log_messages)
        assert any("Loading configuration files" in msg for msg in log_messages)
        assert any("Initializing pipeline components" in msg for msg in log_messages)
        assert any("Pipeline completed successfully" in msg for msg in log_messages)


def test_pipeline_error_handling():
    """Test that pipeline handles errors gracefully."""
    # Test with invalid reporting week format
    result = run_anomaly_detection(
        reporting_week="invalid-date",
        config_path="config/modelconfig.yaml",
        data_config_path="config/dataconfig.yaml",
    )

    # Should return failed status with error details
    assert result["status"] == "failed"
    assert "error" in result
    assert "error_type" in result
    assert isinstance(result["error"], str)

