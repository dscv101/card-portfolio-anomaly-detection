"""Unit tests for DataValidator class."""

from typing import Any

import pandas as pd
import pytest

from src.data.validator import DataValidator
from src.utils.exceptions import ConfigurationError


@pytest.fixture
def validation_config() -> dict[str, Any]:
    """Validation configuration for testing."""
    return {
        "validation": {
            "rules": {
                "missing_critical": {"severity": "CRITICAL", "action": "reject_row"},
                "negative_spend": {"severity": "ERROR", "action": "flag_for_review"},
                "negative_txn_count": {
                    "severity": "ERROR",
                    "action": "flag_for_review",
                },
                "avg_ticket_mismatch": {
                    "severity": "WARNING",
                    "action": "recalculate",
                    "tolerance": 0.01,
                },
                "duplicates": {"severity": "ERROR", "action": "keep_first"},
                "extreme_ticket": {
                    "severity": "WARNING",
                    "action": "flag",
                    "multiplier": 10,
                },
            }
        }
    }


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Valid DataFrame for testing."""
    return pd.DataFrame(
        {
            "customer_id": ["CUST001", "CUST002", "CUST003"],
            "reporting_week": ["2025-11-18", "2025-11-18", "2025-11-18"],
            "mcc": ["5411", "5812", "5999"],
            "spend_amount": [150.0, 200.0, 300.0],
            "transaction_count": [10, 10, 15],
            "avg_ticket_amount": [15.0, 20.0, 20.0],
        }
    )


@pytest.fixture
def dataframe_with_issues() -> pd.DataFrame:
    """DataFrame with various validation issues."""
    return pd.DataFrame(
        {
            "customer_id": [
                "CUST001",
                "CUST002",
                None,
                "CUST004",
                "CUST005",
                "CUST001",
            ],
            "reporting_week": [
                "2025-11-18",
                "2025-11-18",
                "2025-11-18",
                "2025-11-18",
                "2025-11-18",
                "2025-11-18",
            ],
            "mcc": ["5411", "5812", "5999", None, "5411", "5411"],
            "spend_amount": [150.0, -50.0, 100.0, 200.0, 300.0, 150.0],
            "transaction_count": [10, 5, 0, -5, 10, 10],
            "avg_ticket_amount": [15.0, -10.0, 0.0, 40.0, 25.0, 15.0],
        }
    )


class TestDataValidatorInit:
    """Test DataValidator initialization."""

    def test_init_with_valid_config(self, validation_config: dict[str, Any]) -> None:
        """Test initialization with valid configuration."""
        validator = DataValidator(validation_config)
        assert len(validator.rules) == 6
        assert validator.config == validation_config

    def test_init_missing_validation_config(self) -> None:
        """Test initialization fails with missing validation config."""
        with pytest.raises(ConfigurationError, match="Missing 'validation'"):
            DataValidator({})


class TestValidationMethod:
    """Test main validate() method."""

    def test_validate_clean_data(
        self, validation_config: dict[str, Any], valid_dataframe: pd.DataFrame
    ) -> None:
        """Test validation with clean data."""
        validator = DataValidator(validation_config)
        cleaned_df, summary = validator.validate(valid_dataframe)

        assert len(cleaned_df) == 3
        assert summary["total_rows"] == 3
        assert summary["valid_rows"] == 3
        assert summary["rejected_rows"] == 0
        assert len(summary["errors"]) == 0
        assert len(summary["warnings"]) == 0

    def test_validate_data_with_issues(
        self, validation_config: dict[str, Any], dataframe_with_issues: pd.DataFrame
    ) -> None:
        """Test validation with various data issues."""
        validator = DataValidator(validation_config)
        cleaned_df, summary = validator.validate(dataframe_with_issues)

        # Should reject rows with missing critical fields (2 rows: row 2 and row 3)
        assert len(cleaned_df) < len(dataframe_with_issues)
        assert summary["total_rows"] == 6
        assert summary["rejected_rows"] > 0

        # Check that errors were detected
        assert (
            "missing_customer_id" in summary["errors"]
            or "missing_mcc" in summary["errors"]
        )
        assert "negative_spend" in summary["errors"]
        assert "negative_txn_count" in summary["errors"]
        assert "duplicates" in summary["errors"]


class TestMissingCriticalFields:
    """Test missing critical fields detection."""

    def test_missing_customer_id(self, validation_config: dict[str, Any]) -> None:
        """Test detection of missing customer_id."""
        df = pd.DataFrame(
            {
                "customer_id": [None, "CUST002"],
                "reporting_week": ["2025-11-18", "2025-11-18"],
                "mcc": ["5411", "5812"],
                "spend_amount": [100.0, 200.0],
                "transaction_count": [5, 10],
                "avg_ticket_amount": [20.0, 20.0],
            }
        )

        validator = DataValidator(validation_config)
        cleaned_df, summary = validator.validate(df)

        assert summary["errors"]["missing_customer_id"] == 1
        assert len(cleaned_df) == 1  # One row rejected

    def test_missing_reporting_week(self, validation_config: dict[str, Any]) -> None:
        """Test detection of missing reporting_week."""
        df = pd.DataFrame(
            {
                "customer_id": ["CUST001", "CUST002"],
                "reporting_week": [None, "2025-11-18"],
                "mcc": ["5411", "5812"],
                "spend_amount": [100.0, 200.0],
                "transaction_count": [5, 10],
                "avg_ticket_amount": [20.0, 20.0],
            }
        )

        validator = DataValidator(validation_config)
        cleaned_df, summary = validator.validate(df)

        assert summary["errors"]["missing_reporting_week"] == 1
        assert len(cleaned_df) == 1

    def test_missing_mcc(self, validation_config: dict[str, Any]) -> None:
        """Test detection of missing mcc."""
        df = pd.DataFrame(
            {
                "customer_id": ["CUST001", "CUST002"],
                "reporting_week": ["2025-11-18", "2025-11-18"],
                "mcc": [None, "5812"],
                "spend_amount": [100.0, 200.0],
                "transaction_count": [5, 10],
                "avg_ticket_amount": [20.0, 20.0],
            }
        )

        validator = DataValidator(validation_config)
        cleaned_df, summary = validator.validate(df)

        assert summary["errors"]["missing_mcc"] == 1
        assert len(cleaned_df) == 1


class TestNegativeValues:
    """Test negative value detection."""

    def test_negative_spend_amount(self, validation_config: dict[str, Any]) -> None:
        """Test detection of negative spend_amount."""
        df = pd.DataFrame(
            {
                "customer_id": ["CUST001", "CUST002"],
                "reporting_week": ["2025-11-18", "2025-11-18"],
                "mcc": ["5411", "5812"],
                "spend_amount": [100.0, -50.0],
                "transaction_count": [5, 10],
                "avg_ticket_amount": [20.0, -5.0],
            }
        )

        validator = DataValidator(validation_config)
        cleaned_df, summary = validator.validate(df)

        assert summary["errors"]["negative_spend"] == 1
        # Negative values are flagged but not removed
        assert len(cleaned_df) == 2

    def test_negative_transaction_count(
        self, validation_config: dict[str, Any]
    ) -> None:
        """Test detection of negative transaction_count."""
        df = pd.DataFrame(
            {
                "customer_id": ["CUST001", "CUST002"],
                "reporting_week": ["2025-11-18", "2025-11-18"],
                "mcc": ["5411", "5812"],
                "spend_amount": [100.0, 200.0],
                "transaction_count": [5, -10],
                "avg_ticket_amount": [20.0, -20.0],
            }
        )

        validator = DataValidator(validation_config)
        cleaned_df, summary = validator.validate(df)

        assert summary["errors"]["negative_txn_count"] == 1
        assert len(cleaned_df) == 2


class TestDuplicates:
    """Test duplicate detection and handling."""

    def test_duplicate_rows(self, validation_config: dict[str, Any]) -> None:
        """Test detection and removal of duplicate rows."""
        df = pd.DataFrame(
            {
                "customer_id": ["CUST001", "CUST001", "CUST002"],
                "reporting_week": ["2025-11-18", "2025-11-18", "2025-11-18"],
                "mcc": ["5411", "5411", "5812"],
                "spend_amount": [100.0, 150.0, 200.0],
                "transaction_count": [5, 7, 10],
                "avg_ticket_amount": [20.0, 21.43, 20.0],
            }
        )

        validator = DataValidator(validation_config)
        cleaned_df, summary = validator.validate(df)

        assert summary["errors"]["duplicates"] == 1
        assert len(cleaned_df) == 2  # Duplicate removed, first occurrence kept


class TestAvgTicketRecalculation:
    """Test avg_ticket_amount recalculation."""

    def test_avg_ticket_mismatch(self, validation_config: dict[str, Any]) -> None:
        """Test detection and recalculation of mismatched avg_ticket."""
        df = pd.DataFrame(
            {
                "customer_id": ["CUST001", "CUST002"],
                "reporting_week": ["2025-11-18", "2025-11-18"],
                "mcc": ["5411", "5812"],
                "spend_amount": [100.0, 200.0],
                "transaction_count": [5, 10],
                "avg_ticket_amount": [25.0, 15.0],  # Incorrect values
            }
        )

        validator = DataValidator(validation_config)
        cleaned_df, summary = validator.validate(df)

        assert summary["warnings"]["avg_ticket_mismatch"] == 2
        # Values should be recalculated
        assert cleaned_df.iloc[0]["avg_ticket_amount"] == 20.0
        assert cleaned_df.iloc[1]["avg_ticket_amount"] == 20.0

    def test_avg_ticket_division_by_zero(
        self, validation_config: dict[str, Any]
    ) -> None:
        """Test handling of division by zero in avg_ticket calculation."""
        df = pd.DataFrame(
            {
                "customer_id": ["CUST001"],
                "reporting_week": ["2025-11-18"],
                "mcc": ["5411"],
                "spend_amount": [100.0],
                "transaction_count": [0],
                "avg_ticket_amount": [0.0],
            }
        )

        validator = DataValidator(validation_config)
        # Should not raise error, should handle gracefully
        cleaned_df, summary = validator.validate(df)

        assert len(cleaned_df) == 1
        assert cleaned_df.iloc[0]["avg_ticket_amount"] == 0.0


class TestExtremeTickets:
    """Test extreme ticket size detection."""

    def test_extreme_ticket_size(self, validation_config: dict[str, Any]) -> None:
        """Test detection of extreme ticket sizes."""
        df = pd.DataFrame(
            {
                "customer_id": ["CUST001", "CUST002", "CUST003", "CUST004"],
                "reporting_week": [
                    "2025-11-18",
                    "2025-11-18",
                    "2025-11-18",
                    "2025-11-18",
                ],
                "mcc": ["5411", "5411", "5411", "5411"],
                "spend_amount": [100.0, 200.0, 150.0, 5000.0],
                "transaction_count": [5, 10, 8, 10],
                "avg_ticket_amount": [20.0, 20.0, 18.75, 500.0],  # Last one is extreme
            }
        )

        validator = DataValidator(validation_config)
        cleaned_df, summary = validator.validate(df)

        # Median for MCC 5411 is ~19.375, so 500.0 is > 10x median (193.75)
        assert "extreme_ticket_size" in summary["warnings"]
        assert summary["warnings"]["extreme_ticket_size"] >= 1
        assert len(cleaned_df) == 4  # All rows kept, just flagged
