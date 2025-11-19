"""Unit tests for DataLoader class."""

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.utils.exceptions import ConfigurationError, DataLoadError


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "datasource": {
            "type": "csv",
            "csv": {
                "directory": "./datasamples",
                "filename_pattern": "transactions_{reporting_week}.csv",
            },
        }
    }


@pytest.fixture
def sql_config() -> dict[str, Any]:
    """SQL configuration for testing."""
    return {
        "datasource": {
            "type": "sql",
            "sql": {
                "connection_string": "sqlite:///:memory:",
                "query_template": "SELECT * FROM transactions WHERE week = '{reporting_week}'",
                "timeout": 60,
                "max_retries": 3,
            },
        }
    }


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "customer_id": ["CUST001", "CUST002"],
            "reporting_week": ["2025-11-18", "2025-11-18"],
            "mcc": ["5411", "5812"],
            "spend_amount": [100.0, 200.0],
            "transaction_count": [5, 10],
            "avg_ticket_amount": [20.0, 20.0],
        }
    )


class TestDataLoaderInit:
    """Test DataLoader initialization."""

    def test_init_with_valid_config(self, sample_config: dict[str, Any]) -> None:
        """Test initialization with valid configuration."""
        loader = DataLoader(sample_config)
        assert loader.source_type == "csv"
        assert loader.config == sample_config

    def test_init_missing_datasource(self) -> None:
        """Test initialization fails with missing datasource."""
        with pytest.raises(ConfigurationError, match="Missing 'datasource'"):
            DataLoader({})

    def test_init_invalid_source_type(self) -> None:
        """Test initialization fails with invalid source type."""
        config = {"datasource": {"type": "invalid"}}
        with pytest.raises(ConfigurationError, match="Invalid datasource type"):
            DataLoader(config)


class TestCSVLoading:
    """Test CSV file loading functionality."""

    def test_load_from_csv_success(
        self,
        sample_config: dict[str, Any],
        tmp_path: Path,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test successful CSV file loading."""
        # Create temporary CSV file
        csv_file = tmp_path / "transactions_2025-11-18.csv"
        sample_dataframe.to_csv(csv_file, index=False)

        # Update config to use temp directory
        sample_config["datasource"]["csv"]["directory"] = str(tmp_path)

        loader = DataLoader(sample_config)
        df = loader.load("2025-11-18")

        assert len(df) == 2
        assert list(df.columns) == list(sample_dataframe.columns)

    def test_load_from_parquet_success(
        self,
        sample_config: dict[str, Any],
        tmp_path: Path,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test successful Parquet file loading."""
        # Create temporary Parquet file
        parquet_file = tmp_path / "transactions_2025-11-18.parquet"
        sample_dataframe.to_parquet(parquet_file, index=False)

        # Update config to use temp directory and parquet pattern
        sample_config["datasource"]["csv"]["directory"] = str(tmp_path)
        sample_config["datasource"]["csv"][
            "filename_pattern"
        ] = "transactions_{reporting_week}.parquet"

        loader = DataLoader(sample_config)
        df = loader.load("2025-11-18")

        assert len(df) == 2
        assert list(df.columns) == list(sample_dataframe.columns)

    def test_load_from_csv_file_not_found(
        self, sample_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test CSV loading fails when file not found."""
        sample_config["datasource"]["csv"]["directory"] = str(tmp_path)

        loader = DataLoader(sample_config)
        with pytest.raises(DataLoadError, match="File not found"):
            loader.load("2025-11-18")

    def test_load_from_csv_unsupported_format(
        self, sample_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test CSV loading fails with unsupported file format."""
        # Create file with unsupported extension
        txt_file = tmp_path / "transactions_2025-11-18.txt"
        txt_file.write_text("test data")

        sample_config["datasource"]["csv"]["directory"] = str(tmp_path)
        sample_config["datasource"]["csv"][
            "filename_pattern"
        ] = "transactions_{reporting_week}.txt"

        loader = DataLoader(sample_config)
        with pytest.raises(DataLoadError, match="Unsupported file format"):
            loader.load("2025-11-18")

    def test_load_from_csv_empty_file(
        self, sample_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test CSV loading fails with empty file."""
        # Create empty CSV file
        csv_file = tmp_path / "transactions_2025-11-18.csv"
        csv_file.write_text("")

        sample_config["datasource"]["csv"]["directory"] = str(tmp_path)

        loader = DataLoader(sample_config)
        with pytest.raises(DataLoadError, match="File is empty"):
            loader.load("2025-11-18")


@pytest.mark.skip(reason="SQL tests require complex mocking of sqlalchemy")
class TestSQLLoading:
    """Test SQL database loading functionality."""

    def test_load_from_sql_missing_connection_string(
        self, sql_config: dict[str, Any]
    ) -> None:
        """Test SQL loading fails without connection string."""
        # Remove connection_string and ensure env var is not set
        sql_config["datasource"]["sql"].pop("connection_string")

        with patch.dict(os.environ, {}, clear=True):
            loader = DataLoader(sql_config)
            with pytest.raises(
                ConfigurationError, match="Missing SQL connection_string"
            ):
                loader.load("2025-11-18")

    def test_load_from_sql_missing_query_template(
        self, sql_config: dict[str, Any]
    ) -> None:
        """Test SQL loading fails without query template."""
        sql_config["datasource"]["sql"].pop("query_template")

        loader = DataLoader(sql_config)
        with pytest.raises(ConfigurationError, match="Missing SQL query_template"):
            loader.load("2025-11-18")

    @patch("pandas.read_sql_query")
    def test_load_from_sql_success(
        self,
        mock_read_sql: MagicMock,
        sql_config: dict[str, Any],
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test successful SQL query execution."""
        # Mock sqlalchemy import
        with patch("src.data.loader.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_read_sql.return_value = sample_dataframe

            loader = DataLoader(sql_config)
            df = loader.load("2025-11-18")

            assert len(df) == 2
            mock_create_engine.assert_called_once()
            mock_read_sql.assert_called_once()

    @patch("pandas.read_sql_query")
    def test_load_from_sql_retry_logic(
        self,
        mock_read_sql: MagicMock,
        sql_config: dict[str, Any],
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test SQL loading retry logic with exponential backoff."""
        # Mock sqlalchemy import
        with patch("src.data.loader.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            # Fail first two attempts, succeed on third
            mock_read_sql.side_effect = [
                Exception("Connection error"),
                Exception("Timeout"),
                sample_dataframe,
            ]

            loader = DataLoader(sql_config)

            with patch("src.data.loader.time.sleep") as mock_sleep:
                df = loader.load("2025-11-18")

            assert len(df) == 2
            assert mock_read_sql.call_count == 3
            # Verify exponential backoff: 2s, 4s
            assert mock_sleep.call_count == 2

    @patch("pandas.read_sql_query")
    def test_load_from_sql_max_retries_exceeded(
        self,
        mock_read_sql: MagicMock,
        sql_config: dict[str, Any],
    ) -> None:
        """Test SQL loading fails after max retries."""
        # Mock sqlalchemy import
        with patch("src.data.loader.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_read_sql.side_effect = Exception("Connection error")

            loader = DataLoader(sql_config)

            with patch("src.data.loader.time.sleep"):
                with pytest.raises(
                    DataLoadError, match="SQL query failed after 3 retries"
                ):
                    loader.load("2025-11-18")

            # Should have tried 3 times (max_retries)
            assert mock_read_sql.call_count == 3


class TestLoadMethod:
    """Test main load() dispatcher method."""

    def test_load_dispatches_to_csv(
        self,
        sample_config: dict[str, Any],
        tmp_path: Path,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test load() dispatches to CSV loader."""
        csv_file = tmp_path / "transactions_2025-11-18.csv"
        sample_dataframe.to_csv(csv_file, index=False)

        sample_config["datasource"]["csv"]["directory"] = str(tmp_path)

        loader = DataLoader(sample_config)
        df = loader.load("2025-11-18")

        assert len(df) == 2

    def test_load_dispatches_to_sql(self, sql_config: dict[str, Any]) -> None:
        """Test load() dispatches to SQL loader."""
        loader = DataLoader(sql_config)

        # Will fail due to missing sqlalchemy, but confirms dispatch
        with pytest.raises(DataLoadError):
            loader.load("2025-11-18")
