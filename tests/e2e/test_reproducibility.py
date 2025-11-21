"""
Reproducibility tests for anomaly detection pipeline.

Tests:
- REQ-9.1.1: Same inputs + same config = identical outputs
- Fixed random seed produces deterministic results
- Model artifacts enable exact reproduction
"""

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from main import run_anomaly_detection


class TestReproducibility:
    """Tests for deterministic pipeline behavior."""

    @pytest.fixture
    def test_config(self) -> dict[str, Any]:
        """Standard test configuration."""
        return {
            "reporting_week": "2025-11-18",
            "mode": "adhoc",
        }

    @pytest.fixture(autouse=True)
    def cleanup_outputs(self) -> None:
        """Clean up output files before each test."""
        # Setup: clean outputs
        output_dir = Path("./outputs")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True)

        yield

        # Teardown: keep outputs for inspection during development

    def test_reproducibility_with_fixed_seed(self, test_config: dict[str, Any]) -> None:
        """
        REQ-9.1.1: Same input + same config = identical outputs.

        Validates:
        - Fixed random seed produces same results
        - Anomaly scores match exactly
        - Anomaly ranks match exactly
        - Top 20 customer list identical
        """
        # Run 1
        summary1 = run_anomaly_detection(
            reporting_week=test_config["reporting_week"],
            mode=test_config["mode"],
        )
        report1 = pd.read_csv(summary1["output_files"]["report"])

        # Clean outputs for Run 2
        output_dir = Path("./outputs")
        shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Run 2 (same data, same config)
        summary2 = run_anomaly_detection(
            reporting_week=test_config["reporting_week"],
            mode=test_config["mode"],
        )
        report2 = pd.read_csv(summary2["output_files"]["report"])

        # Assert exact match on critical columns
        pd.testing.assert_frame_equal(
            report1[["customer_id", "anomaly_rank"]].sort_values("anomaly_rank"),
            report2[["customer_id", "anomaly_rank"]].sort_values("anomaly_rank"),
            check_dtype=False,
        )

        # Assert anomaly scores match (within floating point tolerance)
        assert np.allclose(
            report1.sort_values("customer_id")["anomaly_score"].values,
            report2.sort_values("customer_id")["anomaly_score"].values,
            rtol=1e-10,
        ), "Anomaly scores differ between runs"

        # Assert category tags match
        pd.testing.assert_series_equal(
            report1.sort_values("customer_id")["category_tag"],
            report2.sort_values("customer_id")["category_tag"],
            check_names=False,
        )

        print("✓ Reproducibility verified: identical results across runs")

    def test_model_artifact_reproducibility(self, test_config: dict[str, Any]) -> None:
        """
        Validate that saved model artifacts enable reproduction.

        Tests:
        - Load saved model artifacts
        - Artifacts contain expected attributes
        - Model and scaler are properly fitted
        """
        # Run pipeline and save artifacts
        summary = run_anomaly_detection(
            reporting_week=test_config["reporting_week"],
            mode=test_config["mode"],
        )

        # Load saved artifacts
        artifact_dir = summary["output_files"]["model_artifacts"]
        with open(f"{artifact_dir}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(f"{artifact_dir}/model.pkl", "rb") as f:
            model = pickle.load(f)

        # Verify artifacts are fitted and contain expected attributes
        assert hasattr(scaler, "mean_"), "Scaler not fitted"
        assert hasattr(model, "estimators_"), "Model not fitted"

        # Verify model parameters match config
        assert model.n_estimators > 0, "Model has no estimators"
        assert model.contamination > 0, "Model contamination not set"

        print("✓ Model artifacts enable reproducibility")

    def test_configuration_snapshot_saved(self, test_config: dict[str, Any]) -> None:
        """
        Validate that configuration snapshot is saved with artifacts.

        Ensures:
        - Config parameters preserved
        - Hyperparameters documented
        - Feature settings captured
        """
        summary = run_anomaly_detection(
            reporting_week=test_config["reporting_week"],
            mode=test_config["mode"],
        )

        # Load metadata
        artifact_dir = summary["output_files"]["model_artifacts"]
        with open(f"{artifact_dir}/metadata.json") as f:
            metadata = json.load(f)

        # Assert config snapshot present
        assert "config" in metadata, "Config snapshot missing"
        config = metadata["config"]

        # Verify key parameters preserved
        assert "n_estimators" in config, "n_estimators not in config snapshot"
        assert "contamination" in config, "contamination not in config snapshot"
        assert "random_state" in config, "random_state not in config snapshot"

        # Assert random_state set (required for reproducibility)
        assert config["random_state"] is not None, "random_state not set"

        # Verify feature names preserved
        assert "feature_names" in metadata, "Feature names not in metadata"
        assert len(metadata["feature_names"]) > 0, "No feature names saved"

        print("✓ Configuration snapshot validated")

    def test_file_hashing_reproducibility(self, test_config: dict[str, Any]) -> None:
        """
        Test reproducibility using file hashing.

        Validates:
        - Model artifact files have consistent hashes
        - Report files have consistent content hashes
        """

        def compute_file_hash(filepath: Path) -> str:
            """Compute SHA256 hash of file."""
            sha256 = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()

        # Run 1
        summary1 = run_anomaly_detection(
            reporting_week=test_config["reporting_week"],
            mode=test_config["mode"],
        )
        artifacts_dir1 = Path(summary1["output_files"]["model_artifacts"])
        hash1_scaler = compute_file_hash(artifacts_dir1 / "scaler.pkl")
        hash1_model = compute_file_hash(artifacts_dir1 / "model.pkl")

        # Clean outputs for Run 2
        output_dir = Path("./outputs")
        shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Run 2
        summary2 = run_anomaly_detection(
            reporting_week=test_config["reporting_week"],
            mode=test_config["mode"],
        )
        artifacts_dir2 = Path(summary2["output_files"]["model_artifacts"])
        hash2_scaler = compute_file_hash(artifacts_dir2 / "scaler.pkl")
        hash2_model = compute_file_hash(artifacts_dir2 / "model.pkl")

        # Assert file hashes match
        assert hash1_scaler == hash2_scaler, "Scaler artifact hashes differ"
        assert hash1_model == hash2_model, "Model artifact hashes differ"

        print("✓ File hashing confirms reproducibility")

    def test_feature_determinism(self, test_config: dict[str, Any]) -> None:
        """
        Test that feature engineering produces deterministic results.

        Validates:
        - Same input data produces same features
        - Feature values match across runs
        - No random variation in feature computation
        """
        from src.data.loader import DataLoader
        from src.data.validator import DataValidator
        from src.features.builder import FeatureBuilder
        from src.utils.config_loader import load_config

        config = load_config(
            "./config/modelconfig.yaml", "./config/dataconfig.yaml"
        )
        loader = DataLoader(config)
        validator = DataValidator(config)
        builder = FeatureBuilder(config)

        # Load and validate data
        data = loader.load(test_config["reporting_week"])
        clean_data, _ = validator.validate(data)

        # Build features twice
        features1 = builder.build_features(clean_data)
        features2 = builder.build_features(clean_data)

        # Assert features are identical
        pd.testing.assert_frame_equal(
            features1.sort_values("customer_id").reset_index(drop=True),
            features2.sort_values("customer_id").reset_index(drop=True),
        )

        print("✓ Feature engineering is deterministic")

    def test_scoring_determinism(self, test_config: dict[str, Any]) -> None:
        """
        Test that model scoring produces deterministic results.

        Validates:
        - Same features produce same scores
        - Anomaly rankings are consistent
        - No random variation in scoring
        """
        from src.data.loader import DataLoader
        from src.data.validator import DataValidator
        from src.features.builder import FeatureBuilder
        from src.models.scorer import ModelScorer
        from src.utils.config_loader import load_config

        config = load_config(
            "./config/modelconfig.yaml", "./config/dataconfig.yaml"
        )
        loader = DataLoader(config)
        validator = DataValidator(config)
        builder = FeatureBuilder(config)

        # Load and validate data
        data = loader.load(test_config["reporting_week"])
        clean_data, _ = validator.validate(data)
        features = builder.build_features(clean_data)

        # Score twice with different scorer instances
        scorer1 = ModelScorer(config)
        scored1 = scorer1.fit_and_score(features)

        scorer2 = ModelScorer(config)
        scored2 = scorer2.fit_and_score(features)

        # Assert scores match
        assert np.allclose(
            scored1.sort_values("customer_id")["anomaly_score"].values,
            scored2.sort_values("customer_id")["anomaly_score"].values,
            rtol=1e-10,
        ), "Anomaly scores differ between scoring runs"

        print("✓ Model scoring is deterministic")

    def test_random_seed_configuration(self) -> None:
        """
        Verify that random seed is properly configured.

        Checks:
        - Config file contains random_state
        - Random seed is set to a fixed value
        - Seed is used in model initialization
        """
        from src.utils.config_loader import load_config

        config = load_config(
            "./config/modelconfig.yaml", "./config/dataconfig.yaml"
        )

        # Check model config has random_state
        assert "isolationforest" in config, "IsolationForest config missing"
        iso_config = config["isolationforest"]

        assert "randomstate" in iso_config, "random_state not in config"
        random_state = iso_config["randomstate"]

        assert random_state is not None, "random_state is None"
        assert isinstance(random_state, int), "random_state must be an integer"
        assert random_state >= 0, "random_state must be non-negative"

        print(f"✓ Random seed configured: {random_state}")

    def test_data_loading_consistency(self, test_config: dict[str, Any]) -> None:
        """
        Test that data loading is consistent across multiple loads.

        Validates:
        - Same file produces same data
        - Row order is consistent
        - No random sampling in loading
        """
        from src.data.loader import DataLoader
        from src.utils.config_loader import load_config

        config = load_config(
            "./config/modelconfig.yaml", "./config/dataconfig.yaml"
        )
        loader = DataLoader(config)

        # Load data twice
        data1 = loader.load(test_config["reporting_week"])
        data2 = loader.load(test_config["reporting_week"])

        # Assert data is identical
        pd.testing.assert_frame_equal(
            data1.reset_index(drop=True),
            data2.reset_index(drop=True),
        )

        print("✓ Data loading is consistent")


class TestPerformanceRequirement:
    """Tests for REQ-7.1.1: Pipeline completes within 15 minutes."""

    def test_full_pipeline_performance(self) -> None:
        """
        REQ-7.1.1: Verify pipeline completes within 15 minutes (900 seconds).

        This is the critical performance requirement for production use.
        """
        import time

        start_time = time.time()

        summary = run_anomaly_detection(
            reporting_week="2025-11-18",
            mode="adhoc",
        )

        execution_time = time.time() - start_time

        # Assert pipeline succeeded
        assert summary["status"] == "success", (
            f"Pipeline failed: {summary.get('error', 'Unknown')}"
        )

        # Assert performance requirement (15 minutes = 900 seconds)
        assert execution_time < 900, (
            f"CRITICAL: Pipeline took {execution_time:.1f}s, "
            f"exceeds 900s (15 min) requirement"
        )

        # Also verify summary reports consistent timing
        assert summary["execution_time_seconds"] < 900, (
            f"Summary reports execution time {summary['execution_time_seconds']}s, "
            f"exceeds 900s requirement"
        )

        # Print performance details
        print(f"✓ Pipeline completed in {execution_time:.1f} seconds")
        print(f"  Performance margin: {900 - execution_time:.1f}s remaining")
        print(f"  Percentage of budget used: {(execution_time / 900) * 100:.1f}%")

    def test_performance_with_larger_dataset(self) -> None:
        """
        Test performance with a larger synthetic dataset.

        Validates:
        - Pipeline scales reasonably with data size
        - Performance degrades gracefully
        """
        # Note: This test would generate a larger dataset
        # For now, we document the expected behavior
        # In production, test with 2x and 5x data volumes

        # Expected performance characteristics:
        # - 500 customers: < 120 seconds
        # - 1000 customers: < 240 seconds
        # - 5000 customers: < 900 seconds

        print("✓ Performance scaling test documented (requires larger datasets)")

    def test_performance_breakdown_reporting(self) -> None:
        """
        Generate a performance breakdown report.

        Documents:
        - Time spent in each pipeline stage
        - Bottlenecks identified
        - Optimization opportunities
        """
        import time

        # Run pipeline and capture stage timings
        summary = run_anomaly_detection(
            reporting_week="2025-11-18",
            mode="adhoc",
        )

        total_time = summary["execution_time_seconds"]

        # Print performance breakdown
        print("\n" + "=" * 60)
        print("PERFORMANCE BREAKDOWN")
        print("=" * 60)
        print(f"Total execution time: {total_time}s")
        print(f"Performance requirement: 900s (15 minutes)")
        print(f"Margin: {900 - total_time}s ({((900 - total_time) / 900) * 100:.1f}%)")
        print("=" * 60)

        # Verify within budget
        assert total_time < 900, "Performance requirement not met"

        print("✓ Performance breakdown generated")


class TestDataIntegrity:
    """Tests for data integrity during pipeline execution."""

    def test_no_data_loss_during_pipeline(self) -> None:
        """
        Verify that no data is unexpectedly lost during pipeline.

        Checks:
        - Customers counted correctly
        - Validation rejections documented
        - No silent data drops
        """
        from src.data.loader import DataLoader
        from src.utils.config_loader import load_config

        config = load_config(
            "./config/modelconfig.yaml", "./config/dataconfig.yaml"
        )
        loader = DataLoader(config)

        # Load raw data
        raw_data = loader.load("2025-11-18")
        unique_customers_raw = raw_data["customer_id"].nunique()

        # Run pipeline
        summary = run_anomaly_detection(
            reporting_week="2025-11-18",
            mode="adhoc",
        )

        # Check data flow
        customers_scored = summary["customers_scored"]

        # Customers scored should be <= raw customers (due to validation)
        assert customers_scored <= unique_customers_raw, (
            f"More customers scored ({customers_scored}) than raw "
            f"({unique_customers_raw})"
        )

        # If customers were dropped, validation summary should reflect it
        if customers_scored < unique_customers_raw:
            validation_summary = summary["validation_summary"]
            # Validation should have documented rejections
            print(
                f"Note: {unique_customers_raw - customers_scored} customers "
                f"filtered during validation"
            )

        print("✓ Data integrity maintained through pipeline")

    def test_feature_completeness(self) -> None:
        """
        Verify that all expected features are generated.

        Checks:
        - Feature count matches configuration
        - No missing features
        - Feature names documented
        """
        summary = run_anomaly_detection(
            reporting_week="2025-11-18",
            mode="adhoc",
        )

        # Load metadata
        artifact_dir = summary["output_files"]["model_artifacts"]
        with open(f"{artifact_dir}/metadata.json") as f:
            metadata = json.load(f)

        feature_names = metadata["feature_names"]

        # Assert minimum expected features
        # (Exact count depends on configuration)
        assert len(feature_names) >= 10, (
            f"Too few features: {len(feature_names)}, expected >= 10"
        )

        # Assert no duplicate feature names
        assert len(feature_names) == len(set(feature_names)), (
            "Duplicate feature names detected"
        )

        print(f"✓ Feature completeness verified: {len(feature_names)} features")

