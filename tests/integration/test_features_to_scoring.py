"""Integration test for features to scoring pipeline.

Tests the complete pipeline from validated data through feature engineering
to model scoring and artifact persistence.

This test ensures:
- FeatureBuilder output is compatible with ModelScorer input
- Scoring pipeline completes without errors
- Model artifacts are saved correctly
- Output DataFrame has expected structure
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.features.builder import FeatureBuilder
from src.models.scorer import ModelScorer


@pytest.fixture
def config():
    """Load configuration from modelconfig.yaml."""
    config_path = Path("config/modelconfig.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_validated_data():
    """Sample validated transaction data (mimics Phase 1 output)."""
    # Simulate customer-week-MCC aggregated data
    return pd.DataFrame(
        {
            "customer_id": ["C001"] * 3 + ["C002"] * 3 + ["C003"] * 3,
            "reporting_week": ["2025-11-18"] * 9,
            "mcc": ["5411", "5812", "5999"] * 3,
            "spend_amount": [1000, 500, 300, 2000, 800, 400, 1500, 600, 350],
            "transaction_count": [10, 5, 3, 20, 8, 4, 15, 6, 3],
        }
    )


def test_features_to_scoring_pipeline(config, sample_validated_data):
    """Test end-to-end pipeline from features to scoring."""
    # Step 1: Build features
    builder = FeatureBuilder(config)
    features_df = builder.build_features(sample_validated_data)

    # Verify features DataFrame structure
    assert not features_df.empty
    assert "customer_id" in features_df.columns
    assert "reporting_week" in features_df.columns
    assert "total_spend" in features_df.columns

    # Step 2: Score anomalies
    scorer = ModelScorer(config)
    scored_df = scorer.fit_and_score(features_df)

    # Verify scored DataFrame structure
    assert len(scored_df) == len(features_df)
    assert "anomaly_score" in scored_df.columns
    assert "anomaly_label" in scored_df.columns

    # Verify anomaly labels are valid
    assert set(scored_df["anomaly_label"].unique()).issubset({-1, 1})

    # Verify original features preserved
    assert "customer_id" in scored_df.columns
    assert "total_spend" in scored_df.columns

    # Step 3: Save artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        reporting_week = "2025-11-18"
        scorer.save_artifacts(tmpdir, reporting_week)

        # Verify artifacts exist
        artifact_dir = Path(tmpdir) / "model_artifacts" / reporting_week
        assert artifact_dir.exists()
        assert (artifact_dir / "scaler.pkl").exists()
        assert (artifact_dir / "model.pkl").exists()
        assert (artifact_dir / "metadata.json").exists()


def test_pipeline_with_larger_dataset(config):
    """Test pipeline with larger, more realistic dataset."""
    # Generate larger synthetic dataset
    import numpy as np

    np.random.seed(42)

    # Create 50 customers with 10 MCCs each
    customers = [f"C{i:04d}" for i in range(50)]
    mccs = [f"{5000 + i}" for i in range(10)]

    data_rows = []
    for customer in customers:
        for mcc in mccs:
            data_rows.append(
                {
                    "customer_id": customer,
                    "reporting_week": "2025-11-18",
                    "mcc": mcc,
                    "spend_amount": np.random.uniform(100, 5000),
                    "transaction_count": np.random.randint(1, 50),
                }
            )

    validated_data = pd.DataFrame(data_rows)

    # Build features
    builder = FeatureBuilder(config)
    features_df = builder.build_features(validated_data)

    assert len(features_df) == 50  # One row per customer

    # Score anomalies
    scorer = ModelScorer(config)
    scored_df = scorer.fit_and_score(features_df)

    # Verify scoring completed
    assert len(scored_df) == 50
    assert "anomaly_score" in scored_df.columns
    assert "anomaly_label" in scored_df.columns

    # Verify anomaly detection rate is reasonable
    anomaly_rate = (scored_df["anomaly_label"] == -1).sum() / len(scored_df)
    expected_contamination = config["isolationforest"]["contamination"]

    # Allow ±100% tolerance for small dataset (statistical variation)
    assert anomaly_rate <= expected_contamination * 2.0


def test_pipeline_handles_edge_cases(config):
    """Test pipeline with edge cases like single customer or minimal data."""
    # Single customer, single MCC
    minimal_data = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "reporting_week": ["2025-11-18"],
            "mcc": ["5411"],
            "spend_amount": [1000],
            "transaction_count": [10],
        }
    )

    builder = FeatureBuilder(config)
    features_df = builder.build_features(minimal_data)

    scorer = ModelScorer(config)
    scored_df = scorer.fit_and_score(features_df)

    # Should complete without errors
    assert len(scored_df) == 1
    assert "anomaly_score" in scored_df.columns
