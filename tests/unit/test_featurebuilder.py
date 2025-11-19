"""Unit tests for FeatureBuilder class.

This module tests all feature engineering functionality including base features,
concentration features, MCC indicators, and delta/growth features.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.builder import FeatureBuilder
from src.utils.exceptions import ConfigurationError, FeatureEngineeringError


@pytest.fixture
def config():
    """Provide standard test configuration."""
    return {
        "features": {
            "top_mcc_count": 3,
            "lookback_windows": [4, 12],
            "concentration_metric": "herfindahl",
            "handle_missing_history": "impute_zero",
        }
    }


@pytest.fixture
def sample_data():
    """Provide sample customer-week-MCC transaction data."""
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C001", "C001", "C002", "C002", "C003"],
            "reporting_week": ["2025-11-18"] * 6,
            "mcc": ["5411", "5812", "5814", "5411", "5912", "5411"],
            "spend_amount": [100.0, 50.0, 25.0, 200.0, 100.0, 150.0],
            "transaction_count": [10, 5, 2, 20, 10, 15],
            "avg_ticket_amount": [10.0, 10.0, 12.5, 10.0, 10.0, 10.0],
        }
    )


@pytest.fixture
def historical_data():
    """Provide historical data for delta feature testing."""
    weeks = []
    for week_offset in range(12, 0, -1):
        week_date = f"2025-{11 - week_offset // 4:02d}-{(week_offset % 4) * 7 + 4:02d}"
        weeks.append(
            {
                "customer_id": "C001",
                "reporting_week": week_date,
                "mcc": "5411",
                "spend_amount": 90.0 + week_offset,
                "transaction_count": 9,
                "avg_ticket_amount": 10.0,
            }
        )
    return pd.DataFrame(weeks)


class TestFeatureBuilderInit:
    """Test FeatureBuilder initialization."""

    def test_init_success(self, config):
        """Test successful initialization with valid config."""
        builder = FeatureBuilder(config)
        assert builder.config == config["features"]

    def test_init_missing_features_key(self):
        """Test initialization fails without 'features' key."""
        with pytest.raises(ConfigurationError, match="Missing 'features'"):
            FeatureBuilder({})


class TestBuildBaseFeatures:
    """Test base feature computation."""

    def test_base_features_computation(self, config, sample_data):
        """Test all base features are computed correctly."""
        builder = FeatureBuilder(config)
        result = builder.build_base_features(sample_data)

        # Check shape and columns
        assert len(result) == 3  # 3 unique customers
        assert list(result.columns) == [
            "customer_id",
            "reporting_week",
            "total_spend",
            "total_transactions",
            "num_active_mccs",
            "avg_ticket_overall",
        ]

        # Check C001 values
        c001 = result[result["customer_id"] == "C001"].iloc[0]
        assert c001["total_spend"] == 175.0  # 100 + 50 + 25
        assert c001["total_transactions"] == 17  # 10 + 5 + 2
        assert c001["num_active_mccs"] == 3  # 5411, 5812, 5814
        assert np.isclose(
            c001["avg_ticket_overall"], 175.0 / 17
        )  # total_spend / total_transactions

    def test_base_features_zero_transactions(self, config):
        """Test handling of zero transactions (avg_ticket should be 0)."""
        data = pd.DataFrame(
            {
                "customer_id": ["C001"],
                "reporting_week": ["2025-11-18"],
                "mcc": ["5411"],
                "spend_amount": [100.0],
                "transaction_count": [0],  # Zero transactions
                "avg_ticket_amount": [0.0],
            }
        )

        builder = FeatureBuilder(config)
        result = builder.build_base_features(data)

        assert result["avg_ticket_overall"].iloc[0] == 0.0

    def test_base_features_single_customer(self, config):
        """Test base features for single customer."""
        data = pd.DataFrame(
            {
                "customer_id": ["C001"],
                "reporting_week": ["2025-11-18"],
                "mcc": ["5411"],
                "spend_amount": [100.0],
                "transaction_count": [10],
                "avg_ticket_amount": [10.0],
            }
        )

        builder = FeatureBuilder(config)
        result = builder.build_base_features(data)

        assert len(result) == 1
        assert result["total_spend"].iloc[0] == 100.0
        assert result["total_transactions"].iloc[0] == 10
        assert result["num_active_mccs"].iloc[0] == 1


class TestBuildConcentrationFeatures:
    """Test concentration feature computation."""

    def test_concentration_features_equal_distribution(self, config):
        """Test concentration with equal MCC distribution (4 MCCs, 25% each)."""
        data = pd.DataFrame(
            {
                "customer_id": ["C001"] * 4,
                "reporting_week": ["2025-11-18"] * 4,
                "mcc": ["5411", "5812", "5814", "5912"],
                "spend_amount": [25.0, 25.0, 25.0, 25.0],  # Equal shares
                "transaction_count": [5, 5, 5, 5],
                "avg_ticket_amount": [5.0, 5.0, 5.0, 5.0],
            }
        )

        builder = FeatureBuilder(config)
        result = builder.build_concentration_features(data)

        # With 4 equal MCCs (25% each):
        # - top_mcc_share = 0.25
        # - top3_mcc_share = 0.75 (3 * 0.25)
        # - herfindahl_index = 4 * (0.25^2) = 0.25
        assert np.isclose(result["top_mcc_share"].iloc[0], 0.25)
        assert np.isclose(result["top3_mcc_share"].iloc[0], 0.75)
        assert np.isclose(result["herfindahl_index"].iloc[0], 0.25)

    def test_concentration_features_single_mcc(self, config):
        """Test concentration with single MCC (100% concentration)."""
        data = pd.DataFrame(
            {
                "customer_id": ["C001"],
                "reporting_week": ["2025-11-18"],
                "mcc": ["5411"],
                "spend_amount": [100.0],
                "transaction_count": [10],
                "avg_ticket_amount": [10.0],
            }
        )

        builder = FeatureBuilder(config)
        result = builder.build_concentration_features(data)

        # Single MCC means 100% concentration
        assert np.isclose(result["top_mcc_share"].iloc[0], 1.0)
        assert np.isclose(result["top3_mcc_share"].iloc[0], 1.0)
        assert np.isclose(result["herfindahl_index"].iloc[0], 1.0)

    def test_concentration_features_dominant_mcc(self, config):
        """Test concentration with one dominant MCC."""
        data = pd.DataFrame(
            {
                "customer_id": ["C001"] * 3,
                "reporting_week": ["2025-11-18"] * 3,
                "mcc": ["5411", "5812", "5814"],
                "spend_amount": [80.0, 10.0, 10.0],  # 80%, 10%, 10%
                "transaction_count": [8, 1, 1],
                "avg_ticket_amount": [10.0, 10.0, 10.0],
            }
        )

        builder = FeatureBuilder(config)
        result = builder.build_concentration_features(data)

        # Dominant MCC (80%)
        assert np.isclose(result["top_mcc_share"].iloc[0], 0.8)
        assert np.isclose(result["top3_mcc_share"].iloc[0], 1.0)  # All 3 MCCs
        # Herfindahl: 0.8^2 + 0.1^2 + 0.1^2 = 0.64 + 0.01 + 0.01 = 0.66
        assert np.isclose(result["herfindahl_index"].iloc[0], 0.66)

    def test_calculate_herfindahl(self, config):
        """Test Herfindahl calculation directly."""
        builder = FeatureBuilder(config)

        # Test equal shares
        shares = pd.Series([0.25, 0.25, 0.25, 0.25])
        assert np.isclose(builder.calculate_herfindahl(shares), 0.25)

        # Test complete concentration
        shares = pd.Series([1.0])
        assert np.isclose(builder.calculate_herfindahl(shares), 1.0)

        # Test known distribution
        shares = pd.Series([0.5, 0.3, 0.2])
        expected = 0.5**2 + 0.3**2 + 0.2**2  # 0.25 + 0.09 + 0.04 = 0.38
        assert np.isclose(builder.calculate_herfindahl(shares), expected)


class TestBuildMCCIndicators:
    """Test MCC indicator feature creation."""

    def test_mcc_indicators_creation(self, config, sample_data):
        """Test binary MCC indicator flags are created correctly."""
        builder = FeatureBuilder(config)
        result = builder.build_mcc_indicators(sample_data)

        # Check columns (should have customer_id, reporting_week + 3 MCC indicators)
        mcc_cols = [col for col in result.columns if col.startswith("mcc_")]
        assert len(mcc_cols) == 3  # top_mcc_count = 3

        # Check that all values are 0 or 1
        for col in mcc_cols:
            assert result[col].isin([0, 1]).all()

        # Check C001 has MCCs (at least 1, since C001 has transactions)
        c001 = result[result["customer_id"] == "C001"].iloc[0]
        assert sum([c001[col] for col in mcc_cols]) >= 1  # Has at least 1 top MCC

    def test_mcc_indicators_top_mccs_selected(self, config):
        """Test that top MCCs by total spend are selected."""
        data = pd.DataFrame(
            {
                "customer_id": ["C001", "C001", "C002", "C002", "C003"],
                "reporting_week": ["2025-11-18"] * 5,
                "mcc": ["5411", "5812", "5411", "5814", "5411"],
                "spend_amount": [100.0, 50.0, 200.0, 25.0, 150.0],
                "transaction_count": [10, 5, 20, 2, 15],
                "avg_ticket_amount": [10.0] * 5,
            }
        )

        builder = FeatureBuilder(config)
        result = builder.build_mcc_indicators(data)

        # Top 3 MCCs by total spend:
        # 5411: 100 + 200 + 150 = 450
        # 5812: 50
        # 5814: 25
        # So indicators should be for 5411, 5812, 5814

        assert "mcc_5411_present" in result.columns
        assert "mcc_5812_present" in result.columns
        assert "mcc_5814_present" in result.columns


class TestBuildDeltaFeatures:
    """Test delta/growth feature computation."""

    def test_delta_features_with_history(self, config, sample_data, historical_data):
        """Test delta features are computed with historical data."""
        builder = FeatureBuilder(config)
        result = builder.build_delta_features(sample_data, historical_data)

        # Check columns
        expected_cols = [
            "customer_id",
            "reporting_week",
            "spend_growth_4w",
            "txn_growth_4w",
            "mcc_diversity_change_4w",
            "spend_growth_12w",
            "mcc_concentration_trend_12w",
        ]
        assert all(col in result.columns for col in expected_cols)

    def test_delta_features_growth_calculation(self, config):
        """Test growth rate calculation with known values."""
        # Current: 150, Historical avg: 100 => growth = (150-100)/100 = 0.5 (50%)
        current_data = pd.DataFrame(
            {
                "customer_id": ["C001"],
                "reporting_week": ["2025-11-18"],
                "mcc": ["5411"],
                "spend_amount": [150.0],
                "transaction_count": [15],
                "avg_ticket_amount": [10.0],
            }
        )

        historical_weeks = []
        for i in range(4):
            historical_weeks.append(
                {
                    "customer_id": "C001",
                    "reporting_week": f"2025-11-{11 + i:02d}",
                    "mcc": "5411",
                    "spend_amount": 100.0,
                    "transaction_count": 10,
                    "avg_ticket_amount": 10.0,
                }
            )
        historical_data = pd.DataFrame(historical_weeks)

        builder = FeatureBuilder(config)
        result = builder.build_delta_features(current_data, historical_data)

        # Growth should be 50%
        assert np.isclose(result["spend_growth_4w"].iloc[0], 0.5)

    def test_delta_features_missing_history_impute_zero(self, config, sample_data):
        """Test missing history handling with impute_zero strategy."""
        # Empty historical data
        historical_data = pd.DataFrame(
            columns=[
                "customer_id",
                "reporting_week",
                "mcc",
                "spend_amount",
                "transaction_count",
                "avg_ticket_amount",
            ]
        )

        builder = FeatureBuilder(config)
        result = builder.build_delta_features(sample_data, historical_data)

        # With impute_zero, missing values should be 0
        assert result["spend_growth_4w"].fillna(0).sum() == 0

    def test_calculate_growth_zero_baseline(self, config):
        """Test growth calculation handles zero baseline."""
        builder = FeatureBuilder(config)

        current = pd.Series([100.0])
        baseline = pd.Series([0.0])  # Zero baseline

        result = builder._calculate_growth(current, baseline, "impute_zero")

        # Should handle gracefully (return 0 with impute_zero)
        assert result.iloc[0] == 0.0

    def test_calculate_trend(self, config):
        """Test linear trend calculation."""
        builder = FeatureBuilder(config)

        # Increasing trend
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        trend = builder._calculate_trend(values)
        assert trend == 1.0  # Slope of 1

        # Decreasing trend
        values = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        trend = builder._calculate_trend(values)
        assert trend == -1.0  # Slope of -1

        # Flat trend
        values = pd.Series([3.0, 3.0, 3.0, 3.0])
        trend = builder._calculate_trend(values)
        assert np.isclose(trend, 0.0)


class TestBuildFeatures:
    """Test end-to-end feature building."""

    def test_build_features_without_historical(self, config, sample_data):
        """Test feature building without historical data (no delta features)."""
        builder = FeatureBuilder(config)
        result = builder.build_features(sample_data)

        # Check shape
        assert len(result) == 3  # 3 unique customers

        # Check base features present
        assert "total_spend" in result.columns
        assert "total_transactions" in result.columns
        assert "num_active_mccs" in result.columns
        assert "avg_ticket_overall" in result.columns

        # Check concentration features present
        assert "top_mcc_share" in result.columns
        assert "top3_mcc_share" in result.columns
        assert "herfindahl_index" in result.columns

        # Check MCC indicators present
        mcc_cols = [col for col in result.columns if col.startswith("mcc_")]
        assert len(mcc_cols) == 3

        # Check NO delta features (no historical data)
        assert "spend_growth_4w" not in result.columns

    def test_build_features_with_historical(self, config, sample_data, historical_data):
        """Test feature building with historical data (includes delta features)."""
        builder = FeatureBuilder(config)
        result = builder.build_features(sample_data, historical_data)

        # All features should be present
        assert "total_spend" in result.columns
        assert "top_mcc_share" in result.columns
        assert "spend_growth_4w" in result.columns
        assert "spend_growth_12w" in result.columns

    def test_build_features_no_nulls_in_critical_features(self, config, sample_data):
        """Test that critical features have no unexpected nulls."""
        builder = FeatureBuilder(config)
        result = builder.build_features(sample_data)

        # Base features should never be null
        assert result["total_spend"].notna().all()
        assert result["total_transactions"].notna().all()
        assert result["num_active_mccs"].notna().all()

        # Concentration features should never be null
        assert result["herfindahl_index"].notna().all()

    def test_build_features_correct_shape(self, config, sample_data):
        """Test feature matrix has correct shape."""
        builder = FeatureBuilder(config)
        result = builder.build_features(sample_data)

        # Should have N customers rows
        n_customers = sample_data["customer_id"].nunique()
        assert len(result) == n_customers

        # Should have multiple feature columns (base + concentration + MCC indicators)
        feature_cols = [
            col
            for col in result.columns
            if col not in ["customer_id", "reporting_week"]
        ]
        assert len(feature_cols) >= 10  # At least 10 features


class TestErrorHandling:
    """Test error handling in FeatureBuilder."""

    def test_build_features_with_invalid_data(self, config):
        """Test feature building fails gracefully with invalid data."""
        builder = FeatureBuilder(config)

        # Empty dataframe
        empty_df = pd.DataFrame()

        with pytest.raises(FeatureEngineeringError):
            builder.build_features(empty_df)

    def test_build_base_features_missing_columns(self, config):
        """Test base features fail with missing required columns."""
        builder = FeatureBuilder(config)

        # Missing 'spend_amount' column
        bad_data = pd.DataFrame(
            {
                "customer_id": ["C001"],
                "reporting_week": ["2025-11-18"],
                "mcc": ["5411"],
                "transaction_count": [10],
            }
        )

        with pytest.raises(FeatureEngineeringError):
            builder.build_base_features(bad_data)
