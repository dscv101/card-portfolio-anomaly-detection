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

    def join_mcc_breakdown(
        self, top_df: pd.DataFrame, raw_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Join detailed MCC breakdown for top N customers.

        Aggregates transaction data by customer and MCC, then joins to the
        top anomalies DataFrame to provide detailed spending patterns.

        Args:
            top_df: DataFrame with top N anomalies
                Required columns: customer_id
            raw_df: Raw transaction detail DataFrame
                Required columns: customer_id, mcc, spend_amount, transaction_count

        Returns:
            DataFrame with MCC breakdown joined to top anomalies.
            Adds columns for each unique MCC with spend and transaction metrics.

        Raises:
            ReportGenerationError: If required columns are missing or join fails
        """
        try:
            # Validate required columns in raw_df
            required_cols = ["customer_id", "mcc", "spend_amount", "transaction_count"]
            missing_cols = [col for col in required_cols if col not in raw_df.columns]
            if missing_cols:
                raise ReportGenerationError(
                    f"raw_df missing required columns: {missing_cols}"
                )

            # Filter raw_df to only include top N customers
            top_customer_ids = top_df["customer_id"].unique()
            filtered_raw = raw_df[raw_df["customer_id"].isin(top_customer_ids)].copy()

            if filtered_raw.empty:
                self.logger.warning(
                    "No matching transaction data found for top customers"
                )
                # Return top_df unchanged
                return top_df

            # Aggregate by customer and MCC
            mcc_agg = (
                filtered_raw.groupby(["customer_id", "mcc"])
                .agg(
                    {
                        "spend_amount": "sum",
                        "transaction_count": "sum",
                    }
                )
                .reset_index()
            )

            # Pivot to create one row per customer with MCC columns
            # Format: mcc_{code}_spend and mcc_{code}_transactions
            mcc_pivot_spend = mcc_agg.pivot(
                index="customer_id", columns="mcc", values="spend_amount"
            )
            mcc_pivot_spend.columns = [f"mcc_{col}_spend" for col in mcc_pivot_spend.columns]

            mcc_pivot_txn = mcc_agg.pivot(
                index="customer_id", columns="mcc", values="transaction_count"
            )
            mcc_pivot_txn.columns = [f"mcc_{col}_transactions" for col in mcc_pivot_txn.columns]

            # Combine spend and transaction pivots
            mcc_combined = pd.concat([mcc_pivot_spend, mcc_pivot_txn], axis=1)
            mcc_combined = mcc_combined.reset_index()

            # Left join to preserve all top customers
            result_df = top_df.merge(mcc_combined, on="customer_id", how="left")

            # Fill NaN with 0 for MCC columns (customers with no transactions in that MCC)
            mcc_cols = [col for col in result_df.columns if col.startswith("mcc_")]
            result_df[mcc_cols] = result_df[mcc_cols].fillna(0)

            self.logger.info(
                f"Joined MCC breakdown for {len(top_df)} customers, "
                f"added {len(mcc_cols)} MCC columns"
            )

            return result_df

        except Exception as e:
            self.logger.error(f"Failed to join MCC breakdown: {e}")
            raise ReportGenerationError(f"Failed to join MCC breakdown: {e}") from e

    def export_csv(
        self, report_df: pd.DataFrame, output_path: str | Path
    ) -> str:
        """Export anomaly report DataFrame to CSV format for Power BI ingestion.

        Args:
            report_df: Final report DataFrame with all columns
                Required columns: customer_id, anomaly_score
            output_path: Path where CSV file should be written

        Returns:
            String path to the exported CSV file

        Raises:
            ReportGenerationError: If export fails
        """
        try:
            output_path = Path(output_path)

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Export to CSV with proper formatting
            report_df.to_csv(
                output_path,
                index=False,
                float_format="%.4f",
                encoding="utf-8",
            )

            file_size = output_path.stat().st_size
            self.logger.info(
                f"Exported CSV report: {output_path} "
                f"({len(report_df)} rows, {file_size:,} bytes)"
            )

            return str(output_path)

        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")
            raise ReportGenerationError(f"Failed to export CSV: {e}") from e

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
