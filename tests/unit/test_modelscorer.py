"""Unit tests for ModelScorer class.

Tests cover:
- Initialization and configuration validation
- Feature matrix preparation and missing value handling
- StandardScaler fitting and transformation
- IsolationForest training
- Anomaly scoring (scores and labels)
- Model artifact persistence
- Error handling edge cases
"""

import json
import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.models.scorer import ModelScorer
from src.utils.exceptions import ConfigurationError, ModelScoringError


@pytest.fixture
def valid_config():
    """Valid ModelScorer configuration."""
    return {
        "isolationforest": {
            "nestimators": 100,
            "contamination": 0.05,
            "maxsamples": 256,
            "randomstate": 42,
            "maxfeatures": 1.0,
            "bootstrap": False,
        }
    }


@pytest.fixture
def sample_features_df():
    """Sample feature DataFrame with numeric features."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "customer_id": [f"CUST{i:04d}" for i in range(100)],
            "reporting_week": ["2025-11-18"] * 100,
            "total_spend": np.random.uniform(100, 10000, 100),
            "total_transactions": np.random.randint(5, 100, 100),
            "avg_ticket": np.random.uniform(10, 500, 100),
            "active_mccs": np.random.randint(1, 15, 100),
            "herfindahl_index": np.random.uniform(0.1, 1.0, 100),
        }
    )


class TestModelScorerInitialization:
    """Test ModelScorer initialization and configuration validation."""

    def test_initialization_with_valid_config(self, valid_config):
        """Test successful initialization with valid configuration."""
        scorer = ModelScorer(valid_config)

        assert scorer.config == valid_config["isolationforest"]
        assert scorer.scaler is None
        assert scorer.model is None
        assert scorer.logger is not None

    def test_initialization_missing_isolationforest_key(self):
        """Test initialization fails without 'isolationforest' key."""
        config: dict[str, Any] = {"features": {}}

        with pytest.raises(
            ConfigurationError, match="Missing 'isolationforest' in configuration"
        ):
            ModelScorer(config)

    def test_initialization_missing_required_params(self):
        """Test initialization fails when required parameters are missing."""
        config = {
            "isolationforest": {
                "nestimators": 100,
                # Missing: contamination, maxsamples, randomstate, maxfeatures
            }
        }

        with pytest.raises(
            ConfigurationError, match="Missing required IsolationForest parameters"
        ):
            ModelScorer(config)


class TestFeaturePreparation:
    """Test feature matrix preparation and missing value handling."""

    def test_prepare_feature_matrix_success(self, valid_config, sample_features_df):
        """Test successful feature matrix preparation."""
        scorer = ModelScorer(valid_config)
        X, feature_cols, valid_mask = scorer.prepare_feature_matrix(sample_features_df)

        # Check shape
        assert X.shape[0] == 100
        assert X.shape[1] == 5  # Exclude customer_id, reporting_week

        # Check feature columns
        assert "total_spend" in feature_cols
        assert "total_transactions" in feature_cols
        assert "customer_id" not in feature_cols
        assert "reporting_week" not in feature_cols

        # Check no NaN values
        assert not np.isnan(X).any()

        # Check valid mask (all rows should be valid)
        assert valid_mask.sum() == 100
        assert valid_mask.all()

    def test_prepare_feature_matrix_with_nan(self, valid_config):
        """Test feature matrix preparation with missing values."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003"],
                "reporting_week": ["2025-11-18"] * 3,
                "total_spend": [1000.0, np.nan, 2000.0],
                "total_transactions": [10, 20, 30],
            }
        )

        scorer = ModelScorer(valid_config)
        X, feature_cols, valid_mask = scorer.prepare_feature_matrix(df)

        # Should drop row with NaN
        assert X.shape[0] == 2
        assert not np.isnan(X).any()

        # Check valid mask (only rows 0 and 2 should be valid)
        assert valid_mask.sum() == 2
        assert valid_mask[0]
        assert not valid_mask[1]  # Row with NaN
        assert valid_mask[2]

    def test_prepare_feature_matrix_all_nan(self, valid_config):
        """Test feature matrix preparation fails when all values are NaN."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002"],
                "reporting_week": ["2025-11-18"] * 2,
                "total_spend": [np.nan, np.nan],
            }
        )

        scorer = ModelScorer(valid_config)
        with pytest.raises(ModelScoringError, match="All feature values are NaN"):
            scorer.prepare_feature_matrix(df)

    def test_prepare_feature_matrix_no_numeric_features(self, valid_config):
        """Test feature matrix preparation fails without numeric features."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002"],
                "reporting_week": ["2025-11-18"] * 2,
                "category": ["A", "B"],
            }
        )

        scorer = ModelScorer(valid_config)
        with pytest.raises(ModelScoringError, match="No numeric features found"):
            scorer.prepare_feature_matrix(df)


class TestFeatureScaling:
    """Test StandardScaler fitting and transformation."""

    def test_fit_scaler(self, valid_config):
        """Test StandardScaler fitting and transformation."""
        scorer = ModelScorer(valid_config)
        np.random.seed(42)
        X = np.random.randn(100, 5) * 10 + 50  # Non-standard scale

        X_scaled = scorer.fit_scaler(X)

        # Check scaler is fitted
        assert scorer.scaler is not None
        assert isinstance(scorer.scaler, StandardScaler)

        # Check scaled features have mean ≈ 0, std ≈ 1
        assert np.abs(X_scaled.mean(axis=0)).max() < 1e-10  # Mean ≈ 0
        assert np.abs(X_scaled.std(axis=0) - 1.0).max() < 1e-10  # Std ≈ 1

        # Check shape preserved
        assert X_scaled.shape == X.shape


class TestModelTraining:
    """Test IsolationForest model training."""

    def test_fit_isolation_forest(self, valid_config):
        """Test IsolationForest fitting with configured parameters."""
        scorer = ModelScorer(valid_config)
        np.random.seed(42)
        X_scaled = np.random.randn(100, 5)

        scorer.fit_isolation_forest(X_scaled)

        # Check model is fitted
        assert scorer.model is not None
        assert isinstance(scorer.model, IsolationForest)

        # Check model parameters (constructor parameters, not fitted)
        assert scorer.model.n_estimators == 100  # type: ignore[attr-defined]
        assert scorer.model.contamination == 0.05  # type: ignore[attr-defined]
        assert scorer.model.max_samples == 256  # type: ignore[attr-defined]
        assert scorer.model.random_state == 42  # type: ignore[attr-defined]

    def test_fit_isolation_forest_with_invalid_data(self, valid_config):
        """Test IsolationForest fitting fails with invalid data."""
        scorer = ModelScorer(valid_config)
        X_invalid = np.array([])  # Empty array

        with pytest.raises(ModelScoringError, match="IsolationForest training failed"):
            scorer.fit_isolation_forest(X_invalid)


class TestAnomalyScoring:
    """Test anomaly score and label generation."""

    def test_score_anomalies(self, valid_config):
        """Test anomaly scoring with fitted model."""
        scorer = ModelScorer(valid_config)
        np.random.seed(42)
        X_scaled = np.random.randn(100, 5)

        # Fit model first
        scorer.fit_isolation_forest(X_scaled)

        # Score anomalies
        scores, labels = scorer.score_anomalies(X_scaled)

        # Check output shapes
        assert scores.shape == (100,)
        assert labels.shape == (100,)

        # Check labels are -1 or 1
        assert set(labels).issubset({-1, 1})

        # Check contamination proportion (should be ≈ 5%)
        anomaly_rate = (labels == -1).sum() / len(labels)
        assert 0.03 <= anomaly_rate <= 0.07  # Allow 3-7% range

    def test_score_anomalies_without_fitted_model(self, valid_config):
        """Test scoring fails without fitted model."""
        scorer = ModelScorer(valid_config)
        np.random.seed(42)
        X_scaled = np.random.randn(100, 5)

        with pytest.raises(ModelScoringError, match="Model not fitted"):
            scorer.score_anomalies(X_scaled)


class TestFitAndScore:
    """Test end-to-end fit_and_score pipeline."""

    def test_fit_and_score_success(self, valid_config, sample_features_df):
        """Test successful end-to-end scoring pipeline."""
        scorer = ModelScorer(valid_config)
        result_df = scorer.fit_and_score(sample_features_df)

        # Check result structure
        assert len(result_df) == 100
        assert "anomaly_score" in result_df.columns
        assert "anomaly_label" in result_df.columns

        # Check original columns preserved
        assert "customer_id" in result_df.columns
        assert "total_spend" in result_df.columns

        # Check anomaly labels
        assert set(result_df["anomaly_label"].unique()).issubset({-1, 1})

        # Check scaler and model are fitted
        assert scorer.scaler is not None
        assert scorer.model is not None

    def test_fit_and_score_empty_dataframe(self, valid_config):
        """Test fit_and_score fails with empty DataFrame."""
        scorer = ModelScorer(valid_config)
        empty_df = pd.DataFrame()

        with pytest.raises(
            ModelScoringError, match="Cannot score empty feature matrix"
        ):
            scorer.fit_and_score(empty_df)

    def test_fit_and_score_reproducibility(self, valid_config, sample_features_df):
        """Test scoring is reproducible with same random state."""
        scorer1 = ModelScorer(valid_config)
        result1 = scorer1.fit_and_score(sample_features_df)

        scorer2 = ModelScorer(valid_config)
        result2 = scorer2.fit_and_score(sample_features_df)

        # Check scores and labels match exactly
        pd.testing.assert_series_equal(
            result1["anomaly_score"], result2["anomaly_score"], check_names=False
        )
        pd.testing.assert_series_equal(
            result1["anomaly_label"], result2["anomaly_label"], check_names=False
        )


class TestArtifactPersistence:
    """Test model artifact saving and loading."""

    def test_save_artifacts_success(self, valid_config, sample_features_df):
        """Test successful artifact persistence."""
        scorer = ModelScorer(valid_config)
        scorer.fit_and_score(sample_features_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            reporting_week = "2025-11-18"
            scorer.save_artifacts(tmpdir, reporting_week)

            # Check directory created
            artifact_dir = Path(tmpdir) / "model_artifacts" / reporting_week
            assert artifact_dir.exists()

            # Check files exist
            assert (artifact_dir / "scaler.pkl").exists()
            assert (artifact_dir / "model.pkl").exists()
            assert (artifact_dir / "metadata.json").exists()

            # Verify scaler can be loaded
            with open(artifact_dir / "scaler.pkl", "rb") as f:
                loaded_scaler = pickle.load(f)
                assert isinstance(loaded_scaler, StandardScaler)

            # Verify model can be loaded
            with open(artifact_dir / "model.pkl", "rb") as f:
                loaded_model = pickle.load(f)
                assert isinstance(loaded_model, IsolationForest)

            # Verify metadata content
            with open(artifact_dir / "metadata.json") as f:
                metadata = json.load(f)
                assert "training_date" in metadata
                assert metadata["reporting_week"] == reporting_week
                assert "config" in metadata
                assert metadata["config"]["nestimators"] == 100

                # Verify training metrics are persisted
                assert "feature_names" in metadata
                assert metadata["feature_names"] is not None
                assert isinstance(metadata["feature_names"], list)
                assert len(metadata["feature_names"]) > 0

                assert "row_count" in metadata
                assert metadata["row_count"] is not None
                assert isinstance(metadata["row_count"], int)
                assert metadata["row_count"] > 0

                assert "anomaly_count" in metadata
                assert metadata["anomaly_count"] is not None
                assert isinstance(metadata["anomaly_count"], int)
                assert metadata["anomaly_count"] >= 0

    def test_save_artifacts_without_fitted_model(self, valid_config):
        """Test artifact saving fails without fitted model."""
        scorer = ModelScorer(valid_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(
                ModelScoringError,
                match="Model and scaler must be fitted before saving artifacts",
            ):
                scorer.save_artifacts(tmpdir, "2025-11-18")

    def test_artifact_versioning(self, valid_config, sample_features_df):
        """Test artifacts are versioned by reporting_week."""
        scorer = ModelScorer(valid_config)
        scorer.fit_and_score(sample_features_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save for week 1
            scorer.save_artifacts(tmpdir, "2025-11-18")

            # Save for week 2
            scorer.save_artifacts(tmpdir, "2025-11-25")

            # Check both directories exist
            artifact_dir1 = Path(tmpdir) / "model_artifacts" / "2025-11-18"
            artifact_dir2 = Path(tmpdir) / "model_artifacts" / "2025-11-25"

            assert artifact_dir1.exists()
            assert artifact_dir2.exists()
            assert (artifact_dir1 / "model.pkl").exists()
            assert (artifact_dir2 / "model.pkl").exists()


class TestNaNHandling:
    """Test NaN handling and shape mismatch bug fix."""

    def test_fit_and_score_with_nan_preserves_shape(self, valid_config):
        """Test that fit_and_score preserves DataFrame shape when NaN present.

        This is a regression test for the critical shape mismatch bug where
        prepare_feature_matrix() drops NaN rows but fit_and_score() tried to
        assign shorter arrays to the full DataFrame.
        """
        # Create DataFrame with some NaN values
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003", "C004", "C005"],
                "reporting_week": ["2025-11-18"] * 5,
                "total_spend": [1000.0, np.nan, 2000.0, 3000.0, np.nan],
                "total_transactions": [10, 20, np.nan, 40, 50],
                "avg_ticket": [100.0, 150.0, 200.0, np.nan, 250.0],
            }
        )

        scorer = ModelScorer(valid_config)
        result_df = scorer.fit_and_score(df)

        # Should preserve original DataFrame shape
        assert len(result_df) == 5

        # Check that anomaly_score and anomaly_label columns exist
        assert "anomaly_score" in result_df.columns
        assert "anomaly_label" in result_df.columns

        # Rows with NaN should have sentinel values
        # Row 0: no NaN -> should have valid score/label
        assert not pd.isna(result_df.loc[0, "anomaly_score"])
        assert result_df.loc[0, "anomaly_label"] in [-1, 1]

        # Row 1: has NaN in total_spend -> should have sentinel values
        assert pd.isna(result_df.loc[1, "anomaly_score"])
        assert result_df.loc[1, "anomaly_label"] == 0

        # Row 2: has NaN in total_transactions -> should have sentinel values
        assert pd.isna(result_df.loc[2, "anomaly_score"])
        assert result_df.loc[2, "anomaly_label"] == 0

        # Row 3: has NaN in avg_ticket -> should have sentinel values
        assert pd.isna(result_df.loc[3, "anomaly_score"])
        assert result_df.loc[3, "anomaly_label"] == 0

        # Row 4: has NaN in total_spend -> should have sentinel values
        assert pd.isna(result_df.loc[4, "anomaly_score"])
        assert result_df.loc[4, "anomaly_label"] == 0

    def test_fit_and_score_with_non_consecutive_index(self, valid_config):
        """Test that the fix works with non-consecutive DataFrame indices.

        This ensures the boolean mask indexing works correctly even when
        DataFrame has gaps in the index (e.g., after filtering operations).
        """
        # Create DataFrame with non-consecutive index
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003", "C004", "C005"],
                "reporting_week": ["2025-11-18"] * 5,
                "total_spend": [1000.0, 2000.0, np.nan, 3000.0, 4000.0],
                "total_transactions": [10, 20, 30, 40, 50],
                "avg_ticket": [100.0, 150.0, 200.0, 250.0, 300.0],
            },
            index=[0, 2, 5, 7, 10],  # Non-consecutive indices
        )

        scorer = ModelScorer(valid_config)
        result_df = scorer.fit_and_score(df)

        # Should preserve original DataFrame shape and indices
        assert len(result_df) == 5
        assert list(result_df.index) == [0, 2, 5, 7, 10]

        # Check sentinel values for row with NaN (index 5)
        assert pd.isna(result_df.loc[5, "anomaly_score"])
        assert result_df.loc[5, "anomaly_label"] == 0

        # Check valid scores for other rows
        for idx in [0, 2, 7, 10]:
            assert not pd.isna(result_df.loc[idx, "anomaly_score"])
            assert result_df.loc[idx, "anomaly_label"] in [-1, 1]

    def test_fit_and_score_all_rows_have_nan(self, valid_config):
        """Test behavior when every row has at least one NaN value.

        This should raise an error since no valid rows remain for training.
        """
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003"],
                "reporting_week": ["2025-11-18"] * 3,
                "total_spend": [np.nan, 2000.0, 3000.0],
                "total_transactions": [10, np.nan, 40],
                "avg_ticket": [100.0, 150.0, np.nan],
            }
        )

        scorer = ModelScorer(valid_config)
        with pytest.raises(ModelScoringError, match="No valid rows remaining"):
            scorer.fit_and_score(df)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_customer(self, valid_config):
        """Test scoring with single customer (edge case but should work)."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001"],
                "reporting_week": ["2025-11-18"],
                "total_spend": [1000.0],
                "total_transactions": [10],
            }
        )

        scorer = ModelScorer(valid_config)
        result_df = scorer.fit_and_score(df)

        assert len(result_df) == 1
        assert "anomaly_score" in result_df.columns
        assert "anomaly_label" in result_df.columns

    def test_high_contamination_warning(self, valid_config, sample_features_df, caplog):
        """Test scoring with high contamination rate logs warning."""
        # Modify config for high contamination
        high_contam_config = valid_config.copy()
        high_contam_config["isolationforest"]["contamination"] = 0.25

        scorer = ModelScorer(high_contam_config)

        # Capture log output
        with caplog.at_level(logging.WARNING):
            result_df = scorer.fit_and_score(sample_features_df)

        # Check warning was logged
        assert any(
            "High contamination value: 0.25 > 0.2" in record.message
            for record in caplog.records
        )

        # Check anomaly rate is approximately 25%
        anomaly_rate = (result_df["anomaly_label"] == -1).sum() / len(result_df)
        assert 0.20 <= anomaly_rate <= 0.30  # Allow 20-30% range
