"""
Unit tests for CLI interface.
"""

from unittest.mock import MagicMock, patch

import pytest

from cli import backtest_command, detect_command, main, validate_command


def test_detect_command_success():
    """Test successful detect command execution."""
    args = MagicMock()
    args.week = "2025-11-18"
    args.mode = "weekly"
    args.model_config = "config/modelconfig.yaml"
    args.data_config = "config/dataconfig.yaml"

    mock_result = {
        "status": "success",
        "execution_time_seconds": 100,
        "data_loaded_rows": 1000,
        "valid_rows": 950,
        "customers_scored": 200,
        "top_anomalies_count": 10,
        "output_files": {
            "report": "outputs/report.csv",
            "summary": "outputs/summary.json",
            "model_artifacts": "outputs/model_artifacts",
        },
    }

    with patch("cli.run_anomaly_detection", return_value=mock_result):
        exit_code = detect_command(args)

    assert exit_code == 0


def test_detect_command_failure():
    """Test detect command with pipeline failure."""
    args = MagicMock()
    args.week = "2025-11-18"
    args.mode = "weekly"
    args.model_config = "config/modelconfig.yaml"
    args.data_config = "config/dataconfig.yaml"

    mock_result = {
        "status": "failed",
        "error": "Configuration error",
        "error_type": "configuration",
    }

    with patch("cli.run_anomaly_detection", return_value=mock_result):
        exit_code = detect_command(args)

    assert exit_code == 2  # Configuration error exit code


def test_detect_command_exit_codes():
    """Test that detect command returns correct exit codes for different errors."""
    args = MagicMock()
    args.week = "2025-11-18"
    args.mode = "weekly"
    args.model_config = "config/modelconfig.yaml"
    args.data_config = "config/dataconfig.yaml"

    error_type_to_exit_code = {
        "configuration": 2,
        "data_validation": 3,
        "model": 4,
        "data_load": 1,
        "unexpected": 1,
    }

    for error_type, expected_exit_code in error_type_to_exit_code.items():
        mock_result = {
            "status": "failed",
            "error": f"{error_type} error",
            "error_type": error_type,
        }

        with patch("cli.run_anomaly_detection", return_value=mock_result):
            exit_code = detect_command(args)

        assert exit_code == expected_exit_code


def test_validate_command_success(tmp_path):
    """Test successful validate command execution."""
    # Create a temporary parquet file
    import pandas as pd

    test_data = pd.DataFrame(
        {
            "customer_id": ["C001", "C002"],
            "reporting_week": ["2025-11-18", "2025-11-18"],
            "mcc": ["5411", "5912"],
            "spend_amount": [100.0, 200.0],
            "transaction_count": [5, 10],
            "avg_ticket_amount": [20.0, 20.0],
        }
    )

    test_file = tmp_path / "test_data.parquet"
    test_data.to_parquet(test_file)

    args = MagicMock()
    args.data_file = str(test_file)
    args.model_config = "config/modelconfig.yaml"
    args.data_config = "config/dataconfig.yaml"

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (
        test_data,
        {
            "critical_failures": 0,
            "errors": 0,
            "warnings": 0,
        },
    )

    with (
        patch("cli.load_config"),
        patch("cli.DataValidator", return_value=mock_validator),
    ):
        exit_code = validate_command(args)

    assert exit_code == 0


def test_validate_command_critical_failures(tmp_path):
    """Test validate command with critical validation failures."""
    import pandas as pd

    test_data = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "reporting_week": ["2025-11-18"],
            "mcc": ["5411"],
            "spend_amount": [100.0],
            "transaction_count": [5],
            "avg_ticket_amount": [20.0],
        }
    )

    test_file = tmp_path / "test_data.parquet"
    test_data.to_parquet(test_file)

    args = MagicMock()
    args.data_file = str(test_file)
    args.model_config = "config/modelconfig.yaml"
    args.data_config = "config/dataconfig.yaml"

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (
        test_data[:0],  # Empty dataframe
        {
            "critical_failures": 5,
            "errors": 2,
            "warnings": 1,
        },
    )

    with (
        patch("cli.load_config"),
        patch("cli.DataValidator", return_value=mock_validator),
    ):
        exit_code = validate_command(args)

    assert exit_code == 3  # Validation failure exit code


def test_validate_command_file_not_found():
    """Test validate command with missing data file."""
    args = MagicMock()
    args.data_file = "nonexistent_file.parquet"
    args.model_config = "config/modelconfig.yaml"
    args.data_config = "config/dataconfig.yaml"

    with patch("cli.load_config"):
        exit_code = validate_command(args)

    assert exit_code == 1


def test_validate_command_unsupported_format(tmp_path):
    """Test validate command with unsupported file format."""
    test_file = tmp_path / "test_data.txt"
    test_file.write_text("some data")

    args = MagicMock()
    args.data_file = str(test_file)
    args.model_config = "config/modelconfig.yaml"
    args.data_config = "config/dataconfig.yaml"

    with patch("cli.load_config"):
        exit_code = validate_command(args)

    assert exit_code == 1


def test_backtest_command_success():
    """Test successful backtest command execution."""
    args = MagicMock()
    args.start_week = "2025-11-01"
    args.end_week = "2025-11-08"  # Two weeks
    args.model_config = "config/modelconfig.yaml"
    args.data_config = "config/dataconfig.yaml"

    mock_result = {
        "status": "success",
        "execution_time_seconds": 100,
        "top_anomalies_count": 10,
    }

    with patch("cli.run_anomaly_detection", return_value=mock_result):
        exit_code = backtest_command(args)

    assert exit_code == 0


def test_backtest_command_with_failures():
    """Test backtest command with some failures."""
    args = MagicMock()
    args.start_week = "2025-11-01"
    args.end_week = "2025-11-08"
    args.model_config = "config/modelconfig.yaml"
    args.data_config = "config/dataconfig.yaml"

    # First call succeeds, second fails
    mock_results = [
        {"status": "success", "execution_time_seconds": 100, "top_anomalies_count": 10},
        {"status": "failed", "error": "Data load error"},
    ]

    with patch("cli.run_anomaly_detection", side_effect=mock_results):
        exit_code = backtest_command(args)

    assert exit_code == 1  # At least one failure


def test_backtest_command_invalid_date_range():
    """Test backtest command with invalid date range."""
    args = MagicMock()
    args.start_week = "2025-11-08"
    args.end_week = "2025-11-01"  # End before start
    args.model_config = "config/modelconfig.yaml"
    args.data_config = "config/dataconfig.yaml"

    exit_code = backtest_command(args)

    assert exit_code == 1


def test_main_detect_command():
    """Test main function with detect command."""
    args = ["detect", "--week", "2025-11-18"]

    mock_result = {
        "status": "success",
        "execution_time_seconds": 100,
        "data_loaded_rows": 1000,
        "valid_rows": 950,
        "customers_scored": 200,
        "top_anomalies_count": 10,
        "output_files": {
            "report": "outputs/report.csv",
            "summary": "outputs/summary.json",
            "model_artifacts": "outputs/model_artifacts",
        },
    }

    with patch("cli.run_anomaly_detection", return_value=mock_result):
        exit_code = main(args)

    assert exit_code == 0


def test_main_validate_command(tmp_path):
    """Test main function with validate command."""
    import pandas as pd

    test_data = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "reporting_week": ["2025-11-18"],
            "mcc": ["5411"],
            "spend_amount": [100.0],
            "transaction_count": [5],
            "avg_ticket_amount": [20.0],
        }
    )

    test_file = tmp_path / "test_data.parquet"
    test_data.to_parquet(test_file)

    args = ["validate", "--data-file", str(test_file)]

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (
        test_data,
        {"critical_failures": 0, "errors": 0, "warnings": 0},
    )

    with (
        patch("cli.load_config"),
        patch("cli.DataValidator", return_value=mock_validator),
    ):
        exit_code = main(args)

    assert exit_code == 0


def test_main_backtest_command():
    """Test main function with backtest command."""
    args = ["backtest", "--start-week", "2025-11-01", "--end-week", "2025-11-08"]

    mock_result = {
        "status": "success",
        "execution_time_seconds": 100,
        "top_anomalies_count": 10,
    }

    with patch("cli.run_anomaly_detection", return_value=mock_result):
        exit_code = main(args)

    assert exit_code == 0


def test_main_no_command():
    """Test main function without a command."""
    args: list[str] = []

    exit_code = main(args)

    assert exit_code == 1


def test_main_unknown_command():
    """Test main function with unknown command."""
    args = ["unknown"]

    exit_code = main(args)

    assert exit_code == 1


def test_main_help():
    """Test main function with help flag."""
    args = ["--help"]

    with pytest.raises(SystemExit) as exc_info:
        main(args)

    assert exc_info.value.code == 0
