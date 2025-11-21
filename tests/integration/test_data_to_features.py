"""Integration tests for data loading → feature engineering pipeline.
This module tests the end-to-end flow from raw data to feature matrix.
"""

import pandas as pd
import pytest
import yaml

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.features.builder import FeatureBuilder


@pytest.fixture
def config(tmp_path):
    """Load configuration from YAML files."""
    with open("config/dataconfig.yaml") as f:
        data_config = yaml.safe_load(f)
    with open("config/modelconfig.yaml") as f:
        model_config = yaml.safe_load(f)
    # Override for testing with CSV source
    data_config["datasource"]["type"] = "csv"
    data_config["datasource"]["csv"] = {
        "directory": str(tmp_path),
        "filename_pattern": "transactions_{reporting_week}.csv",
    }
    return {"data": data_config, "model": model_config}


@pytest.fixture
def sample_csv_data(tmp_path):
    """Create sample CSV data file for testing."""
    data = pd.DataFrame(
        {
            "customer_id": ["C001", "C001", "C001", "C002", "C002", "C003"] * 5,
            "reporting_week": ["2025-11-18"] * 30,
            "mcc": (["5411", "5812", "5814"] * 10),
            "spend_amount": [100.0, 50.0, 25.0, 200.0, 100.0, 150.0] * 5,
            "transaction_count": [10, 5, 2, 20, 10, 15] * 5,
            "avg_ticket_amount": [10.0, 10.0, 12.5, 10.0, 10.0, 10.0] * 5,
        }
    )
    # Save to CSV with expected filename pattern
    csv_path = tmp_path / "transactions_2025-11-18.csv"
    data.to_csv(csv_path, index=False)
    return str(tmp_path)


class TestDataToFeaturesPipeline:
    """Test end-to-end pipeline from data loading to feature engineering."""

    def test_pipeline_executes_without_errors(self, config, sample_csv_data):
        """Test complete pipeline executes without errors."""
        # Step 1: Load data
        loader = DataLoader(config["data"])
        raw_data = loader.load("2025-11-18")
        assert raw_data is not None
        assert len(raw_data) > 0
        # Step 2: Validate data
        validator = DataValidator(config["data"])
        clean_data, validation_report = validator.validate(raw_data)
        assert clean_data is not None
        assert len(clean_data) > 0
        # Step 3: Build features
        builder = FeatureBuilder(config["model"])
        features = builder.build_features(clean_data)
        assert features is not None
        assert len(features) > 0

    def test_pipeline_feature_matrix_shape(self, config, sample_csv_data):
        """Test feature matrix has expected shape."""
        # Execute pipeline
        loader = DataLoader(config["data"])
        validator = DataValidator(config["data"])
        builder = FeatureBuilder(config["model"])
        raw_data = loader.load("2025-11-18")
        clean_data, _ = validator.validate(raw_data)
        features = builder.build_features(clean_data)
        # Check shape
        n_customers = clean_data["customer_id"].nunique()
        assert len(features) == n_customers
        # Check feature columns exist
        expected_features = [
            "customer_id",
            "reporting_week",
            "total_spend",
            "total_transactions",
            "avg_ticket_overall",
            "num_active_mccs",
            "top_mcc_share",
            "top3_mcc_share",
            "herfindahl_index",
        ]
        for feature in expected_features:
            assert feature in features.columns

    def test_pipeline_no_unexpected_nulls(self, config, sample_csv_data):
        """Test feature matrix has no unexpected null values."""
        # Execute pipeline
        loader = DataLoader(config["data"])
        validator = DataValidator(config["data"])
        builder = FeatureBuilder(config["model"])
        raw_data = loader.load("2025-11-18")
        clean_data, _ = validator.validate(raw_data)
        features = builder.build_features(clean_data)
        # Check critical features have no nulls
        critical_features = [
            "total_spend",
            "total_transactions",
            "num_active_mccs",
            "herfindahl_index",
        ]
        for feature in critical_features:
            null_count: int = features[feature].isnull().sum()
            assert (
                null_count == 0
            ), f"Feature '{feature}' has {null_count} unexpected nulls"

    def test_pipeline_feature_distributions_reasonable(self, config, sample_csv_data):
        """Test feature distributions are reasonable (no NaN, Inf)."""
        # Execute pipeline
        loader = DataLoader(config["data"])
        validator = DataValidator(config["data"])
        builder = FeatureBuilder(config["model"])
        raw_data = loader.load("2025-11-18")
        clean_data, _ = validator.validate(raw_data)
        features = builder.build_features(clean_data)
        # Check no infinite values
        numeric_cols = features.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            assert (
                not features[col].isin([float("inf"), float("-inf")]).any()
            ), f"Feature '{col}' contains infinite values"

    def test_pipeline_with_historical_data(self, config, sample_csv_data, tmp_path):
        """Test pipeline with historical data for delta features."""
        # Create historical data
        historical_data = pd.DataFrame(
            {
                "customer_id": ["C001"] * 48,  # 12 weeks * 4 MCCs
                "reporting_week": [
                    f"2025-{10 + i // 16:02d}-{(i % 16) + 1:02d}" for i in range(48)
                ],
                "mcc": ["5411", "5812", "5814", "5912"] * 12,
                "spend_amount": [90.0, 45.0, 20.0, 15.0] * 12,
                "transaction_count": [9, 4, 2, 1] * 12,
                "avg_ticket_amount": [10.0] * 48,
            }
        )
        # Execute pipeline
        loader = DataLoader(config["data"])
        validator = DataValidator(config["data"])
        builder = FeatureBuilder(config["model"])
        raw_data = loader.load("2025-11-18")
        clean_data, _ = validator.validate(raw_data)
        # Validate historical data
        clean_historical, _ = validator.validate(historical_data)
        # Build features with historical data
        features = builder.build_features(clean_data, clean_historical)
        # Check delta features are present
        delta_features = [
            "spend_growth_4w",
            "txn_growth_4w",
            "mcc_diversity_change_4w",
            "spend_growth_12w",
            "mcc_concentration_trend_12w",
        ]
        for feature in delta_features:
            assert feature in features.columns, f"Delta feature '{feature}' missing"

    def test_pipeline_performance(self, config, sample_csv_data):
        """Test pipeline completes within reasonable time for small dataset."""
        import time

        # Execute pipeline and measure time
        start_time = time.time()
        loader = DataLoader(config["data"])
        validator = DataValidator(config["data"])
        builder = FeatureBuilder(config["model"])
        raw_data = loader.load("2025-11-18")
        clean_data, _ = validator.validate(raw_data)
        features = builder.build_features(clean_data)
        elapsed = time.time() - start_time
        # Should complete in reasonable time (< 5 seconds for small dataset)
        assert elapsed < 5.0, f"Pipeline took {elapsed:.2f}s (expected < 5s)"
        # Log timing for reference
        print(f"\nPipeline execution time: {elapsed:.2f}s")
        print(
            f"Processed {len(raw_data)} input rows → {len(features)} customer features"
        )


class TestFeatureQuality:
    """Test quality and correctness of computed features."""

    def test_base_features_match_raw_data(self, config, sample_csv_data):
        """Test base features aggregate raw data correctly."""
        # Execute pipeline
        loader = DataLoader(config["data"])
        validator = DataValidator(config["data"])
        builder = FeatureBuilder(config["model"])
        raw_data = loader.load("2025-11-18")
        clean_data, _ = validator.validate(raw_data)
        features = builder.build_features(clean_data)
        # Manually calculate expected totals for one customer
        c001_data = clean_data[clean_data["customer_id"] == "C001"]
        expected_spend = c001_data["spend_amount"].sum()
        expected_txns = c001_data["transaction_count"].sum()
        # Check computed features match
        c001_features = features[features["customer_id"] == "C001"].iloc[0]
        assert c001_features["total_spend"] == expected_spend
        assert c001_features["total_transactions"] == expected_txns

    def test_concentration_features_bounds(self, config, sample_csv_data):
        """Test concentration features are within valid bounds."""
        # Execute pipeline
        loader = DataLoader(config["data"])
        validator = DataValidator(config["data"])
        builder = FeatureBuilder(config["model"])
        raw_data = loader.load("2025-11-18")
        clean_data, _ = validator.validate(raw_data)
        features = builder.build_features(clean_data)
        # Herfindahl index should be in [0, 1]
        assert (features["herfindahl_index"] >= 0).all()
        assert (features["herfindahl_index"] <= 1).all()
        # Top MCC share should be in [0, 1]
        assert (features["top_mcc_share"] >= 0).all()
        assert (features["top_mcc_share"] <= 1).all()
        # Top 3 MCC share should be in [0, 1]
        assert (features["top3_mcc_share"] >= 0).all()
        assert (features["top3_mcc_share"] <= 1).all()

    def test_mcc_indicators_are_binary(self, config, sample_csv_data):
        """Test MCC indicator features are binary (0 or 1)."""
        # Execute pipeline
        loader = DataLoader(config["data"])
        validator = DataValidator(config["data"])
        builder = FeatureBuilder(config["model"])
        raw_data = loader.load("2025-11-18")
        clean_data, _ = validator.validate(raw_data)
        features = builder.build_features(clean_data)
        # Get MCC indicator columns
        mcc_cols = [
            col
            for col in features.columns
            if col.startswith("mcc_") and col.endswith("_present")
        ]
        # Check all values are 0 or 1
        for col in mcc_cols:
            assert (
                features[col].isin([0, 1]).all()
            ), f"MCC indicator '{col}' has non-binary values"
