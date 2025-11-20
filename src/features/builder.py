"""Feature engineering module for transforming transaction data into features.

This module provides the FeatureBuilder class which computes base features,
concentration metrics, MCC indicators, and delta/growth features from validated
customer-week-MCC transaction data.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.exceptions import ConfigurationError, FeatureEngineeringError

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Transform validated transaction data into feature matrix.

    The FeatureBuilder computes features in a structured pipeline:
    1. Base features (spend, transactions, avg_ticket, active_mccs)
    2. Concentration features (Herfindahl index, top MCC shares)
    3. MCC indicator features (binary flags for top N MCCs)
    4. Delta/growth features (optional, requires historical data)

    Attributes:
        config: Feature configuration from modelconfig.yaml
        logger: Logger instance for tracking feature computation
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with modelconfig.yaml feature settings.

        Args:
            config: Configuration dictionary containing 'features' key

        Raises:
            ConfigurationError: If required configuration is missing
        """
        if "features" not in config:
            raise ConfigurationError("Missing 'features' in configuration")

        self.config = config["features"]
        self.logger = logging.getLogger("features.builder")
        self.logger.info("FeatureBuilder initialized")

    def build_features(
        self,
        df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Transform validated transaction data into feature matrix.

        This method orchestrates the feature computation pipeline. Features
        are computed in a specific order to ensure proper dependencies.

        Args:
            df: Current week validated DataFrame with columns:
                customer_id, reporting_week, mcc, spend_amount,
                transaction_count, avg_ticket_amount
            historical_df: Optional 12-week lookback DataFrame for delta features

        Returns:
            DataFrame with columns:
            - customer_id, reporting_week (identifiers)
            - base features (total_spend, total_transactions, etc.)
            - concentration features (herfindahl_index, top_mcc_share, etc.)
            - mcc indicators (mcc_XXXX_present binary flags)
            - delta features (if historical_df provided)

        Raises:
            FeatureEngineeringError: If feature computation fails
        """
        try:
            self.logger.info(f"Building features for {len(df)} input rows")

            # Step 1: Build base features (aggregated to customer-week level)
            features_df = self.build_base_features(df)
            self.logger.info(f"Built base features for {len(features_df)} customers")

            # Step 2: Build concentration features
            concentration_df = self.build_concentration_features(df)
            features_df = features_df.merge(
                concentration_df,
                on=["customer_id", "reporting_week"],
                how="left",
            )
            self.logger.info("Built concentration features")

            # Step 3: Build MCC indicator features
            mcc_indicators_df = self.build_mcc_indicators(df)
            features_df = features_df.merge(
                mcc_indicators_df,
                on=["customer_id", "reporting_week"],
                how="left",
            )
            self.logger.info("Built MCC indicator features")

            # Step 4: Build delta/growth features (optional)
            if historical_df is not None:
                delta_df = self.build_delta_features(df, historical_df)
                features_df = features_df.merge(
                    delta_df,
                    on=["customer_id", "reporting_week"],
                    how="left",
                )
                self.logger.info("Built delta/growth features")
            else:
                self.logger.info(
                    "Skipping delta features (no historical data provided)"
                )

            # Log feature summary
            feature_cols = [
                col
                for col in features_df.columns
                if col not in ["customer_id", "reporting_week"]
            ]
            null_pct = features_df[feature_cols].isnull().sum() / len(features_df) * 100
            self.logger.info(
                f"Built {len(feature_cols)} features for {len(features_df)} customers"
            )
            self.logger.debug(f"Null percentages:\n{null_pct}")

            return features_df

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise FeatureEngineeringError(f"Failed to build features: {str(e)}") from e

    def build_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute base features from customer-week-MCC transaction detail.

        Aggregates MCC-level detail to customer-week level and computes:
        - total_spend: Sum of spend across all MCCs
        - total_transactions: Sum of transaction counts across all MCCs
        - avg_ticket_overall: Total spend / total transactions
        - num_active_mccs: Count of distinct MCCs per customer

        Args:
            df: Input DataFrame with customer_id, reporting_week, mcc,
                spend_amount, transaction_count

        Returns:
            DataFrame with customer_id, reporting_week, and base features

        Raises:
            FeatureEngineeringError: If aggregation fails
        """
        try:
            # Aggregate to customer-week level
            base_features = (
                df.groupby(["customer_id", "reporting_week"])
                .agg(
                    total_spend=("spend_amount", "sum"),
                    total_transactions=("transaction_count", "sum"),
                    num_active_mccs=("mcc", "nunique"),
                )
                .reset_index()
            )

            # Calculate average ticket overall
            # Handle zero transactions: set avg_ticket to 0
            base_features["avg_ticket_overall"] = np.where(
                base_features["total_transactions"] > 0,
                base_features["total_spend"] / base_features["total_transactions"],
                0.0,
            )

            self.logger.debug(
                f"Base features computed for {len(base_features)} customer-weeks"
            )

            return base_features

        except Exception as e:
            raise FeatureEngineeringError(
                f"Failed to compute base features: {str(e)}"
            ) from e

    def build_concentration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute concentration features using MCC spend distribution.

        Calculates:
        - top_mcc_share: Percentage of spend in largest MCC
        - top3_mcc_share: Percentage of spend in top 3 MCCs
        - herfindahl_index: Sum of squared spend shares (H = Σ(shareᵢ²))

        The Herfindahl index measures concentration:
        - Range: [1/N, 1] where N = number of MCCs
        - Perfect diversification (equal shares): H = 1/N
        - Complete concentration (single MCC): H = 1.0

        Args:
            df: Input DataFrame with customer_id, reporting_week, mcc, spend_amount

        Returns:
            DataFrame with customer_id, reporting_week, and concentration features

        Raises:
            FeatureEngineeringError: If concentration calculation fails
        """
        try:
            # Calculate total spend per customer-week for share calculation
            customer_totals = (
                df.groupby(["customer_id", "reporting_week"])["spend_amount"]
                .sum()
                .reset_index()
                .rename(columns={"spend_amount": "total_spend"})
            )

            # Join to get spend share for each MCC
            df_with_totals = df.merge(
                customer_totals,
                on=["customer_id", "reporting_week"],
            )
            df_with_totals["mcc_share"] = (
                df_with_totals["spend_amount"] / df_with_totals["total_spend"]
            )

            # Calculate top MCC share (max share)
            top_mcc_share = (
                df_with_totals.groupby(["customer_id", "reporting_week"])["mcc_share"]
                .max()
                .reset_index()
                .rename(columns={"mcc_share": "top_mcc_share"})
            )

            # Calculate top 3 MCC share
            top3_mcc_share = (
                df_with_totals.sort_values("mcc_share", ascending=False)
                .groupby(["customer_id", "reporting_week"])
                .head(3)
                .groupby(["customer_id", "reporting_week"])["mcc_share"]
                .sum()
                .reset_index()
                .rename(columns={"mcc_share": "top3_mcc_share"})
            )

            # Calculate Herfindahl index (sum of squared shares)
            herfindahl = (
                df_with_totals.groupby(["customer_id", "reporting_week"])
                .apply(
                    lambda x: self.calculate_herfindahl(x["mcc_share"]),
                    include_groups=False,
                )
                .reset_index()
                .rename(columns={0: "herfindahl_index"})
            )

            # Merge all concentration features
            concentration_features = top_mcc_share.merge(
                top3_mcc_share, on=["customer_id", "reporting_week"]
            ).merge(herfindahl, on=["customer_id", "reporting_week"])

            self.logger.debug(
                f"Concentration features computed for "
                f"{len(concentration_features)} customer-weeks"
            )

            return concentration_features

        except Exception as e:
            raise FeatureEngineeringError(
                f"Failed to compute concentration features: {str(e)}"
            ) from e

    def calculate_herfindahl(self, spend_shares: pd.Series) -> float:
        """Calculate Herfindahl index (sum of squared spend shares).

        The Herfindahl index is a measure of concentration calculated as:
        H = Σᵢ(shareᵢ²)

        where shareᵢ is the proportion of spend in MCC i.

        Args:
            spend_shares: Series of MCC spend shares (values between 0 and 1)

        Returns:
            Herfindahl index value (between 1/N and 1, where N is number of MCCs)
        """
        return float((spend_shares**2).sum())

    def build_mcc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary flags for top N MCCs.

        Identifies the top N MCCs by total spend across the entire portfolio
        and creates binary indicator columns (mcc_XXXX_present) for each customer.

        Args:
            df: Input DataFrame with customer_id, reporting_week, mcc, spend_amount

        Returns:
            DataFrame with customer_id, reporting_week, and binary MCC indicators

        Raises:
            FeatureEngineeringError: If MCC indicator creation fails
        """
        try:
            top_mcc_count = self.config.get("top_mcc_count", 10)

            # Identify top N MCCs by total spend across portfolio
            top_mccs = (
                df.groupby("mcc")["spend_amount"]
                .sum()
                .nlargest(top_mcc_count)
                .index.tolist()
            )

            self.logger.info(f"Top {top_mcc_count} MCCs: {top_mccs}")

            # Get unique customer-week combinations
            mcc_indicators = (
                df[["customer_id", "reporting_week"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            # Create binary indicator for each top MCC
            for mcc_code in top_mccs:
                # Get customers who have this MCC
                customers_with_mcc = df[df["mcc"] == mcc_code][
                    ["customer_id", "reporting_week"]
                ].drop_duplicates()
                customers_with_mcc[f"mcc_{mcc_code}_present"] = 1

                # Merge and fill missing with 0
                mcc_indicators = mcc_indicators.merge(
                    customers_with_mcc,
                    on=["customer_id", "reporting_week"],
                    how="left",
                )
                mcc_indicators[f"mcc_{mcc_code}_present"] = (
                    mcc_indicators[f"mcc_{mcc_code}_present"].fillna(0).astype(int)
                )

            self.logger.debug(
                f"Created MCC indicators for {len(mcc_indicators)} customer-weeks"
            )

            return mcc_indicators

        except Exception as e:
            raise FeatureEngineeringError(
                f"Failed to create MCC indicators: {str(e)}"
            ) from e

    def build_delta_features(
        self,
        df: pd.DataFrame,
        historical_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute growth features comparing current week to historical averages.

        Calculates:
        - spend_growth_4w: 4-week spend growth rate
        - txn_growth_4w: 4-week transaction growth rate
        - mcc_diversity_change_4w: Change in number of active MCCs (4-week)
        - spend_growth_12w: 12-week spend growth rate
        - mcc_concentration_trend_12w: Trend in Herfindahl index (12-week)

        Growth rate formula: (current - baseline) / baseline
        Handles missing history based on config (impute_zero, drop_feature, flag_only).

        Args:
            df: Current week DataFrame
            historical_df: Historical DataFrame (12-week lookback)

        Returns:
            DataFrame with customer_id, reporting_week, and delta features

        Raises:
            FeatureEngineeringError: If delta feature calculation fails
        """
        try:
            lookback_windows = self.config.get("lookback_windows", [4, 12])
            handle_missing = self.config.get("handle_missing_history", "impute_zero")

            # Build base features for current and historical data
            current_base = self.build_base_features(df)
            historical_base = self.build_base_features(historical_df)

            # Initialize delta features dataframe
            delta_features = current_base[["customer_id", "reporting_week"]].copy()

            # Calculate 4-week averages
            if 4 in lookback_windows:
                historical_4w = (
                    historical_base.sort_values(["customer_id", "reporting_week"])
                    .groupby("customer_id")
                    .tail(4)
                    .groupby("customer_id")
                    .agg(
                        avg_spend_4w=("total_spend", "mean"),
                        avg_txn_4w=("total_transactions", "mean"),
                        avg_mccs_4w=("num_active_mccs", "mean"),
                    )
                    .reset_index()
                )

                # Merge with current data
                delta_features = delta_features.merge(
                    current_base[
                        [
                            "customer_id",
                            "reporting_week",
                            "total_spend",
                            "total_transactions",
                            "num_active_mccs",
                        ]
                    ],
                    on=["customer_id", "reporting_week"],
                ).merge(historical_4w, on="customer_id", how="left")

                # Calculate growth rates
                delta_features["spend_growth_4w"] = self._calculate_growth(
                    delta_features["total_spend"],
                    delta_features["avg_spend_4w"],
                    handle_missing,
                )
                delta_features["txn_growth_4w"] = self._calculate_growth(
                    delta_features["total_transactions"],
                    delta_features["avg_txn_4w"],
                    handle_missing,
                )
                delta_features["mcc_diversity_change_4w"] = (
                    delta_features["num_active_mccs"] - delta_features["avg_mccs_4w"]
                ).fillna(0 if handle_missing == "impute_zero" else np.nan)

                # Drop intermediate columns
                delta_features = delta_features.drop(
                    columns=[
                        "total_spend",
                        "total_transactions",
                        "num_active_mccs",
                        "avg_spend_4w",
                        "avg_txn_4w",
                        "avg_mccs_4w",
                    ]
                )

            # Calculate 12-week averages and trends
            if 12 in lookback_windows:
                historical_12w = (
                    historical_base.sort_values(["customer_id", "reporting_week"])
                    .groupby("customer_id")
                    .tail(12)
                    .groupby("customer_id")
                    .agg(avg_spend_12w=("total_spend", "mean"))
                    .reset_index()
                )

                # Merge for 12-week spend growth
                temp_df = current_base[
                    ["customer_id", "reporting_week", "total_spend"]
                ].merge(historical_12w, on="customer_id", how="left")

                delta_features["spend_growth_12w"] = self._calculate_growth(
                    temp_df["total_spend"],
                    temp_df["avg_spend_12w"],
                    handle_missing,
                )

                # Calculate concentration trend (slope of Herfindahl over 12 weeks)
                # Only compute if we have historical data
                if len(historical_df) > 0:
                    # Build concentration for historical data
                    historical_concentration = self.build_concentration_features(
                        historical_df
                    )
                    concentration_trend = (
                        historical_concentration.groupby("customer_id")
                        .apply(
                            lambda x: self._calculate_trend(x["herfindahl_index"]),
                            include_groups=False,
                        )
                        .reset_index()
                        .rename(columns={0: "mcc_concentration_trend_12w"})
                    )

                    delta_features = delta_features.merge(
                        concentration_trend,
                        on="customer_id",
                        how="left",
                    )
                    delta_features["mcc_concentration_trend_12w"] = delta_features[
                        "mcc_concentration_trend_12w"
                    ].fillna(0 if handle_missing == "impute_zero" else np.nan)
                else:
                    # No historical data - impute based on config
                    delta_features["mcc_concentration_trend_12w"] = (
                        0.0 if handle_missing == "impute_zero" else np.nan
                    )

            self.logger.debug(
                f"Delta features computed for {len(delta_features)} customer-weeks"
            )

            return delta_features

        except Exception as e:
            raise FeatureEngineeringError(
                f"Failed to compute delta features: {str(e)}"
            ) from e

    def _calculate_growth(
        self,
        current: pd.Series,
        baseline: pd.Series,
        handle_missing: str,
    ) -> pd.Series:
        """Calculate growth rate: (current - baseline) / baseline.

        Handles zero baseline and missing values based on configuration.

        Args:
            current: Current period values
            baseline: Baseline (historical average) values
            handle_missing: How to handle missing/zero baseline ('impute_zero', etc.)

        Returns:
            Growth rate series
        """
        # Avoid division by zero
        growth = np.where(
            (baseline > 0) & baseline.notna(),
            (current - baseline) / baseline,
            np.nan,
        )

        if handle_missing == "impute_zero":
            growth = np.nan_to_num(growth, nan=0.0)

        return pd.Series(growth, index=current.index)

    def _calculate_trend(self, values: pd.Series) -> float:
        """Calculate linear trend (slope) of values over time.

        Uses simple linear regression: y = mx + b, returns m (slope).

        Args:
            values: Time series values

        Returns:
            Slope of linear trend
        """
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = values.values

        # Simple linear regression
        x_mean = x.mean()
        y_mean = y.mean()

        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return float(slope)
