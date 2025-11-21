"""Data validation module for enforcing data quality rules.

This module provides the DataValidator class which validates DataFrames
against configured rules and generates validation summaries.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class DataValidator:
    """Enforce data quality rules and log violations.

    The validator checks for missing values, negative amounts, duplicates,
    and other data quality issues based on REQ-3.2.1 specifications.

    Attributes:
        config: Configuration dictionary from dataconfig.yaml
        rules: Validation rules configuration
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with validation rules.

        Args:
            config: Configuration dictionary containing validation settings

        Raises:
            ConfigurationError: If validation configuration is missing
        """
        if "validation" not in config:
            raise ConfigurationError("Missing 'validation' in configuration")

        self.config = config
        self.rules = config["validation"].get("rules", {})

        logger.info(f"DataValidator initialized with {len(self.rules)} rules")

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Validate DataFrame against rules.

        Applies all validation rules and generates a validation summary with
        counts of errors, warnings, and cleaned data statistics.

        Args:
            df: Input DataFrame to validate

        Returns:
            Tuple of:
                - cleaned_df: DataFrame with invalid rows removed/fixed
                - validation_summary: Dict with counts, warnings, errors
        """
        logger.info(f"Starting validation of {len(df)} rows")

        total_rows = len(df)
        errors: dict[str, int] = {}
        warnings: dict[str, int] = {}

        # Create a copy to avoid modifying original
        cleaned_df = df.copy()

        # 1. Check for negative values FIRST (before filtering rows)
        # This ensures we detect all negative values even in rows that will be rejected
        negative_counts = self._check_negative_values(cleaned_df)
        if negative_counts:
            errors.update(negative_counts)
            # Flag but don't remove - keep for review
            logger.warning(f"Found {sum(negative_counts.values())} negative values")

        # 2. Check for missing critical fields and remove them
        missing_counts = self._check_missing_critical(cleaned_df)
        if missing_counts:
            errors.update(missing_counts)
            # Remove rows with missing critical fields
            cleaned_df = cleaned_df.dropna(
                subset=["customer_id", "reporting_week", "mcc"]
            )
            logger.error(
                f"{sum(missing_counts.values())} rows rejected due to "
                f"missing critical fields"
            )

        # 3. Check for duplicates
        duplicate_count = self._check_duplicates(cleaned_df)
        if duplicate_count > 0:
            errors["duplicates"] = duplicate_count
            # Keep first occurrence
            cleaned_df = cleaned_df.drop_duplicates(
                subset=["customer_id", "reporting_week", "mcc"], keep="first"
            )
            logger.error(f"{duplicate_count} duplicate rows found, keeping first")

        # 4. Recalculate and check avg_ticket_amount
        mismatch_count = self._recalculate_avg_ticket(cleaned_df, warnings)
        if mismatch_count > 0:
            warnings["avg_ticket_mismatch"] = mismatch_count
            logger.warning(
                f"{mismatch_count} rows had avg_ticket mismatch, recalculated"
            )

        # 5. Flag extreme ticket sizes
        extreme_count = self._flag_extreme_tickets(cleaned_df)
        if extreme_count > 0:
            warnings["extreme_ticket_size"] = extreme_count
            logger.warning(f"{extreme_count} rows flagged for extreme ticket sizes")

        valid_rows = len(cleaned_df)
        rejected_rows = total_rows - valid_rows

        validation_summary = {
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "rejected_rows": rejected_rows,
            "errors": errors,
            "warnings": warnings,
        }

        logger.info(
            f"Validation complete: {valid_rows}/{total_rows} rows valid, "
            f"{rejected_rows} rejected"
        )

        return cleaned_df, validation_summary

    def _check_missing_critical(self, df: pd.DataFrame) -> dict[str, int]:
        """Check for missing critical fields (customer_id, reporting_week, mcc).

        Args:
            df: DataFrame to check

        Returns:
            Dict with counts of missing values per critical field
        """
        missing_counts = {}
        critical_fields = ["customer_id", "reporting_week", "mcc"]

        for field in critical_fields:
            if field in df.columns:
                missing_count: int = int(df[field].isna().sum())
                if missing_count > 0:
                    missing_counts[f"missing_{field}"] = missing_count
                    logger.error(f"Found {missing_count} missing values in {field}")

        return missing_counts

    def _check_negative_values(self, df: pd.DataFrame) -> dict[str, int]:
        """Check for negative spend_amount and transaction_count.

        Args:
            df: DataFrame to check

        Returns:
            Dict with counts of negative values per field
        """
        negative_counts = {}

        if "spend_amount" in df.columns:
            negative_spend: int = int((df["spend_amount"] < 0).sum())
            if negative_spend > 0:
                negative_counts["negative_spend"] = negative_spend
                logger.error(f"Found {negative_spend} negative spend values")

        if "transaction_count" in df.columns:
            negative_txn: int = int((df["transaction_count"] < 0).sum())
            if negative_txn > 0:
                negative_counts["negative_txn_count"] = negative_txn
                logger.error(f"Found {negative_txn} negative transaction_count values")

        return negative_counts

    def _check_duplicates(self, df: pd.DataFrame) -> int:
        """Check for duplicate (customer_id, reporting_week, mcc) combinations.

        Args:
            df: DataFrame to check

        Returns:
            Count of duplicate rows
        """
        if all(col in df.columns for col in ["customer_id", "reporting_week", "mcc"]):
            duplicates = df.duplicated(
                subset=["customer_id", "reporting_week", "mcc"], keep="first"
            )
            duplicate_count = duplicates.sum()
            if duplicate_count > 0:
                logger.error(f"Found {duplicate_count} duplicate rows")
            return int(duplicate_count)
        return 0

    def _recalculate_avg_ticket(
        self, df: pd.DataFrame, warnings: dict[str, int]
    ) -> int:
        """Recalculate avg_ticket_amount where it doesn't match spend/transaction_count.

        Args:
            df: DataFrame to check and fix (modified in place)
            warnings: Dict to update with warning counts

        Returns:
            Count of rows with mismatched avg_ticket
        """
        tolerance = self.rules.get("avg_ticket_mismatch", {}).get("tolerance", 0.01)

        if all(
            col in df.columns
            for col in ["spend_amount", "transaction_count", "avg_ticket_amount"]
        ):
            # Calculate expected avg_ticket
            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                expected_avg = df["spend_amount"] / df["transaction_count"]
                expected_avg = expected_avg.fillna(0)

            # Find mismatches (absolute difference > tolerance)
            mismatch_mask = np.abs(df["avg_ticket_amount"] - expected_avg) > tolerance

            # Exclude rows where transaction_count is 0
            mismatch_mask = mismatch_mask & (df["transaction_count"] != 0)

            mismatch_count = mismatch_mask.sum()

            if mismatch_count > 0:
                # Recalculate for mismatched rows
                df.loc[mismatch_mask, "avg_ticket_amount"] = expected_avg[mismatch_mask]
                logger.warning(
                    f"Recalculated avg_ticket_amount for {mismatch_count} rows"
                )

            return int(mismatch_count)

        return 0

    def _flag_extreme_tickets(self, df: pd.DataFrame) -> int:
        """Flag rows where avg_ticket is > 10x MCC median.

        Args:
            df: DataFrame to check

        Returns:
            Count of rows flagged for extreme ticket sizes
        """
        multiplier = self.rules.get("extreme_ticket", {}).get("multiplier", 10)

        if "mcc" in df.columns and "avg_ticket_amount" in df.columns:
            # Calculate median avg_ticket per MCC
            mcc_medians = df.groupby("mcc")["avg_ticket_amount"].median()

            # Map medians back to DataFrame
            df["_mcc_median"] = df["mcc"].map(mcc_medians)

            # Flag extreme values
            extreme_mask = df["avg_ticket_amount"] > (multiplier * df["_mcc_median"])
            extreme_count = extreme_mask.sum()

            if extreme_count > 0:
                logger.warning(
                    f"Flagged {extreme_count} rows with extreme ticket sizes "
                    f"(>{multiplier}x MCC median)"
                )

            # Clean up temporary column
            df.drop(columns=["_mcc_median"], inplace=True)

            return int(extreme_count)

        return 0
