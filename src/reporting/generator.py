"""Report generation module for anomaly detection output.

This module provides the ReportGenerator class which handles:
- Ranking customers by anomaly score
- Selecting top N anomalies for reporting
- Applying rule-based category tags
- Joining detailed MCC breakdown for top customers
- Exporting CSV reports for Power BI ingestion
- Exporting JSON summaries with execution metadata
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.exceptions import ReportGenerationError

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate anomaly reports and summary metadata.

    The ReportGenerator implements the complete reporting pipeline:
    1. Rank customers by anomaly_score (ascending - most anomalous first)
    2. Select top N customers (configurable, default 20)
    3. Apply rule-based category tags (REQ-6.2.1)
    4. Join detailed MCC breakdown for top N customers
    5. Export CSV report for Power BI
    6. Export JSON summary with execution metadata

    Attributes:
        config: Reporting configuration from modelconfig.yaml
        logger: Logger instance for tracking reporting operations
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with reporting config from modelconfig.yaml.

        Args:
            config: Dictionary containing reporting configuration with keys:
                - reporting.topnanomalies: Number of top anomalies to report (default 20)
                - reporting.rules: Business rules for filtering/tagging

        Raises:
            ReportGenerationError: If required configuration is missing
        """
        if "reporting" not in config:
            raise ReportGenerationError("Missing 'reporting' in configuration")

        self.config = config["reporting"]
        self.logger = logging.getLogger("reporting.generator")

        # Validate required configuration keys
        if "topnanomalies" not in self.config:
            raise ReportGenerationError(
                "Missing 'topnanomalies' in reporting configuration"
            )

        self.logger.info(
            f"ReportGenerator initialized with top_n={self.config['topnanomalies']}"
        )

    def generate(
        self, scored_df: pd.DataFrame, raw_df: pd.DataFrame, reporting_week: str
    ) -> tuple[str, str]:
        """Generate anomaly report and summary metadata.

        Args:
            scored_df: DataFrame with features + anomaly scores
                Required columns: customer_id, anomaly_score, anomaly_label
            raw_df: Original transaction detail for MCC breakdown
                Required columns: customer_id, mcc, spend_amount, transaction_count
            reporting_week: Reporting week in YYYY-MM-DD format

        Returns:
            Tuple of (report_path, summary_path):
                - report_path: Path to anomaly_report_{YYYY-MM-DD}.csv
                - summary_path: Path to anomaly_summary_{YYYY-MM-DD}.json

        Raises:
            ReportGenerationError: If required columns are missing or generation fails

        Process:
            1. Rank customers by anomaly_score (ascending)
            2. Select top N (configurable, default 20)
            3. Apply rule-based tags (REQ-6.2.1)
            4. Join detailed MCC breakdown for top N
            5. Export CSV + JSON
        """
        try:
            # Validate input DataFrames
            self._validate_inputs(scored_df, raw_df)

            # Create output directory if it doesn't exist
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            # Define output paths
            report_path = output_dir / f"anomaly_report_{reporting_week}.csv"
            summary_path = output_dir / f"anomaly_summary_{reporting_week}.json"

            self.logger.info(
                f"Starting report generation for reporting_week={reporting_week}"
            )

            # Placeholder for full pipeline
            # Will be implemented in subsequent tasks:
            # 1. rank_anomalies()
            # 2. apply_tags()
            # 3. join_mcc_breakdown()
            # 4. export_csv()
            # 5. export_summary_json()

            self.logger.info(f"Report generation completed for {reporting_week}")
            self.logger.info(f"CSV report: {report_path}")
            self.logger.info(f"JSON summary: {summary_path}")

            return (str(report_path), str(summary_path))

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise ReportGenerationError(f"Failed to generate report: {e}") from e

    def rank_anomalies(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Rank customers by anomaly score (ascending - most anomalous first).

        Sorts the scored DataFrame by anomaly_score in ascending order.
        Lower (more negative) scores indicate stronger anomalies.

        Args:
            scored_df: DataFrame with anomaly scores
                Required columns: customer_id, anomaly_score

        Returns:
            DataFrame sorted by anomaly_score (ascending)

        Raises:
            ReportGenerationError: If required columns are missing
        """
        try:
            # Validate required columns
            if "anomaly_score" not in scored_df.columns:
                raise ReportGenerationError(
                    "scored_df missing required column: anomaly_score"
                )

            # Sort by anomaly_score ascending (most anomalous first)
            ranked_df = scored_df.sort_values(
                by="anomaly_score", ascending=True
            ).reset_index(drop=True)

            self.logger.info(
                f"Ranked {len(ranked_df)} customers by anomaly score"
            )
            self.logger.debug(
                f"Top anomaly score: {ranked_df['anomaly_score'].iloc[0]:.4f}"
            )

            return ranked_df

        except Exception as e:
            self.logger.error(f"Failed to rank anomalies: {e}")
            raise ReportGenerationError(f"Failed to rank anomalies: {e}") from e

    def apply_tags(self, ranked_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
        """Select top N anomalies and apply rule-based category tags.

        Selects the top N most anomalous customers and applies business rule
        tags based on configuration (REQ-6.2.1).

        Args:
            ranked_df: DataFrame sorted by anomaly_score (ascending)
                Required columns: customer_id, anomaly_score
                Optional columns used for tagging: total_spend, total_transactions
            top_n: Number of top anomalies to select

        Returns:
            DataFrame with top N anomalies and additional tag columns:
                - meets_min_spend: bool, True if spend >= minspend threshold
                - meets_min_transactions: bool, True if transactions >= mintransactions
                - rule_flagged: bool, True if any rule threshold not met

        Raises:
            ReportGenerationError: If required columns are missing or top_n invalid
        """
        try:
            # Validate top_n parameter
            if top_n <= 0:
                raise ReportGenerationError(
                    f"top_n must be positive, got {top_n}"
                )

            if top_n > len(ranked_df):
                self.logger.warning(
                    f"top_n ({top_n}) exceeds DataFrame length ({len(ranked_df)}), "
                    f"using all {len(ranked_df)} rows"
                )
                top_n = len(ranked_df)

            # Select top N anomalies
            top_df = ranked_df.head(top_n).copy()

            # Apply business rule tags if configuration exists
            rules = self.config.get("rules", {})
            
            # Initialize tag columns with default values
            top_df["meets_min_spend"] = True
            top_df["meets_min_transactions"] = True
            top_df["rule_flagged"] = False

            # Apply minspend rule if configured and column exists
            if "minspend" in rules and "total_spend" in top_df.columns:
                min_spend = rules["minspend"]
                top_df["meets_min_spend"] = top_df["total_spend"] >= min_spend
                self.logger.debug(f"Applied minspend rule: threshold={min_spend}")

            # Apply mintransactions rule if configured and column exists
            if "mintransactions" in rules and "total_transactions" in top_df.columns:
                min_txns = rules["mintransactions"]
                top_df["meets_min_transactions"] = (
                    top_df["total_transactions"] >= min_txns
                )
                self.logger.debug(
                    f"Applied mintransactions rule: threshold={min_txns}"
                )

            # Set rule_flagged for any customer failing rules
            top_df["rule_flagged"] = ~(
                top_df["meets_min_spend"] & top_df["meets_min_transactions"]
            )

            flagged_count = top_df["rule_flagged"].sum()
            self.logger.info(
                f"Selected top {len(top_df)} anomalies, {flagged_count} flagged by rules"
            )

            return top_df

        except Exception as e:
            self.logger.error(f"Failed to apply tags: {e}")
            raise ReportGenerationError(f"Failed to apply tags: {e}") from e

    def _validate_inputs(self, scored_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
        """Validate input DataFrames have required columns.

        Args:
            scored_df: Scored features DataFrame
            raw_df: Raw transaction DataFrame

        Raises:
            ReportGenerationError: If required columns are missing
        """
        # Required columns in scored_df
        required_scored_cols = ["customer_id", "anomaly_score", "anomaly_label"]
        missing_scored = [
            col for col in required_scored_cols if col not in scored_df.columns
        ]
        if missing_scored:
            raise ReportGenerationError(
                f"scored_df missing required columns: {missing_scored}"
            )

        # Required columns in raw_df
        required_raw_cols = [
            "customer_id",
            "mcc",
            "spend_amount",
            "transaction_count",
        ]
        missing_raw = [col for col in required_raw_cols if col not in raw_df.columns]
        if missing_raw:
            raise ReportGenerationError(
                f"raw_df missing required columns: {missing_raw}"
            )

        # Validate non-empty DataFrames
        if scored_df.empty:
            raise ReportGenerationError("scored_df is empty")
        if raw_df.empty:
            raise ReportGenerationError("raw_df is empty")

        self.logger.debug(
            f"Input validation passed: {len(scored_df)} scored customers, "
            f"{len(raw_df)} raw transactions"
        )
