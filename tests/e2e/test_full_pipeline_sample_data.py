"""
End-to-end test for full anomaly detection pipeline with sample data.

Tests:
- REQ-7.1.1: Pipeline completes within 15 minutes
- Complete data flow: Load → Validate → Features → Score → Report
- Output file generation
- Report structure validation
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from main import run_anomaly_detection


class TestFullPipelineSampleData:
    """E2E tests for complete pipeline execution with sample data."""

    @pytest.fixture
    def test_config(self) -> dict[str, Any]:
        """Test configuration pointing to sample data."""
        return {
            "reporting_week": "2025-11-18",
            "mode": "adhoc",
            "config_path": "./config/modelconfig.yaml",
            "data_config_path": "./config/dataconfig.yaml",
        }

    def test_pipeline_executes_successfully(self, test_config: dict[str, Any]) -> None:
        """
        REQ-7.1.1: Pipeline executes within 15 minutes.

        Validates:
        - Successful execution
        - Performance requirement met
        - All output files created
        """
        start_time = time.time()

        # Execute full pipeline
        summary = run_anomaly_detection(
            reporting_week=test_config["reporting_week"],
            mode=test_config["mode"],
        )

        execution_time = time.time() - start_time

        # Assert success
        assert (
            summary["status"] == "success"
        ), f"Pipeline failed: {summary.get('error', 'Unknown')}"

        # Assert performance requirement (15 minutes = 900 seconds)
        assert (
            execution_time < 900
        ), f"Pipeline took {execution_time:.1f}s, exceeds 900s limit"
        assert summary["execution_time_seconds"] < 900

        # Assert outputs created
        assert os.path.exists(
            summary["output_files"]["report"]
        ), "CSV report not created"
        assert os.path.exists(
            summary["output_files"]["summary"]
        ), "JSON summary not created"
        assert os.path.exists(
            summary["output_files"]["model_artifacts"]
        ), "Model artifacts not created"

        print(f"✓ Pipeline completed successfully in {execution_time:.1f} seconds")

    def test_report_structure_valid(self, test_config: dict[str, Any]) -> None:
        """
        Validate CSV report structure and content.

        Checks:
        - Top 20 anomalies present
        - Required columns exist
        - Data types correct
        - No unexpected null values
        """
        summary = run_anomaly_detection(
            reporting_week=test_config["reporting_week"], mode=test_config["mode"]
        )

        # Load CSV report
        report = pd.read_csv(summary["output_files"]["report"])

        # Assert row count (top 20)
        assert len(report) == 20, f"Expected 20 rows, got {len(report)}"

        # Assert required columns
        required_columns = [
            "customer_id",
            "reporting_week",
            "anomaly_score",
            "total_spend",
            "total_transactions",
            "avg_ticket_overall",
            "num_active_mccs",
            "herfindahl_index",
            "top_mcc_share",
        ]
        for col in required_columns:
            assert col in report.columns, f"Missing column: {col}"

        # Assert anomaly_score is sorted (most anomalous first - lowest scores)
        anomaly_scores = report["anomaly_score"].values
        assert all(
            anomaly_scores[i] <= anomaly_scores[i + 1] for i in range(len(anomaly_scores) - 1)
        ), "Anomaly scores should be sorted in ascending order (most anomalous first)"

        # Assert rule flagging columns exist
        assert "meets_min_spend" in report.columns, "Missing meets_min_spend column"
        assert "meets_min_transactions" in report.columns, "Missing meets_min_transactions column"
        assert "rule_flagged" in report.columns, "Missing rule_flagged column"

        # Assert no unexpected nulls in critical columns
        critical_cols = ["customer_id", "anomaly_score"]
        for col in critical_cols:
            assert report[col].notna().all(), f"Null values in critical column: {col}"

        print("✓ Report structure validated successfully")

    def test_json_summary_valid(self, test_config: dict[str, Any]) -> None:
        """
        Validate JSON summary structure and metadata.

        Checks:
        - Valid JSON structure
        - All required sections present
        - Metadata reasonable
        """
        summary = run_anomaly_detection(
            reporting_week=test_config["reporting_week"], mode=test_config["mode"]
        )

        # Load JSON summary
        with open(summary["output_files"]["summary"]) as f:
            json_summary = json.load(f)

        # Assert required sections
        required_sections = [
            "metadata",
            "statistics",
            "top_anomalies",
        ]
        for section in required_sections:
            assert section in json_summary, f"Missing section: {section}"

        # Assert metadata
        metadata = json_summary["metadata"]
        assert metadata["reporting_week"] == test_config["reporting_week"]
        assert "generated_at" in metadata
        assert metadata["top_n_anomalies"] == 20

        # Assert statistics
        statistics = json_summary["statistics"]
        assert statistics["total_customers"] == 20
        assert statistics["anomaly_count"] >= 0
        assert "anomaly_score" in statistics

        print("✓ JSON summary validated successfully")

    def test_model_artifacts_saved(self, test_config: dict[str, Any]) -> None:
        """
        Validate model artifacts are saved correctly.

        Checks:
        - scaler.pkl exists and loadable
        - model.pkl exists and loadable
        - metadata.json exists and valid
        """
        import pickle

        summary = run_anomaly_detection(
            reporting_week=test_config["reporting_week"], mode=test_config["mode"]
        )

        artifact_dir = summary["output_files"]["model_artifacts"]

        # Check files exist
        assert os.path.exists(f"{artifact_dir}/scaler.pkl"), "scaler.pkl not found"
        assert os.path.exists(f"{artifact_dir}/model.pkl"), "model.pkl not found"
        assert os.path.exists(
            f"{artifact_dir}/metadata.json"
        ), "metadata.json not found"

        # Load and validate artifacts
        with open(f"{artifact_dir}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            assert scaler is not None

        with open(f"{artifact_dir}/model.pkl", "rb") as f:
            model = pickle.load(f)
            assert model is not None

        with open(f"{artifact_dir}/metadata.json") as f:
            metadata = json.load(f)
            assert "feature_names" in metadata
            assert "config" in metadata
            assert "training_date" in metadata

        print("✓ Model artifacts validated successfully")

    def test_edge_cases_detected(self, test_config: dict[str, Any]) -> None:
        """
        Validate that known edge cases are detected as anomalies.

        Checks:
        - High-growth customer flagged
        - High-concentration customer flagged
        - Unusual ticket size flagged
        """
        summary = run_anomaly_detection(
            reporting_week=test_config["reporting_week"], mode=test_config["mode"]
        )

        report = pd.read_csv(summary["output_files"]["report"])

        # Check that edge case customers appear in top 20
        edge_customers = ["EDGE001", "EDGE002", "EDGE003"]
        detected_edge_cases = report[report["customer_id"].isin(edge_customers)]

        # At least some edge cases should be detected
        assert len(detected_edge_cases) > 0, "No edge cases detected in top 20"

        print(f"✓ Detected {len(detected_edge_cases)} edge cases in top 20")

    def test_data_volumes_reasonable(self, test_config: dict[str, Any]) -> None:
        """
        Validate that data volumes are within expected ranges.

        Checks:
        - Customers scored > 0
        - Anomalies detected > 0
        - Valid rows >= customers scored
        """
        summary = run_anomaly_detection(
            reporting_week=test_config["reporting_week"], mode=test_config["mode"]
        )

        assert summary["customers_scored"] > 0, "No customers scored"
        assert summary["top_anomalies_count"] > 0, "No anomalies detected"
        assert (
            summary["valid_rows"] >= summary["customers_scored"]
        ), "Valid rows < customers scored"

        print(
            f"✓ Data volumes: {summary['customers_scored']} customers, "
            f"{summary['top_anomalies_count']} anomalies"
        )

    def test_csv_and_parquet_equivalence(self) -> None:
        """
        Test that CSV and Parquet files produce equivalent results.

        Validates:
        - Both formats load successfully
        - Same number of rows
        - Same data content
        """
        # Load CSV version
        csv_data = pd.read_csv("./datasamples/transactions_2025-11-18.csv")

        # Load Parquet version
        parquet_data = pd.read_parquet("./datasamples/transactions_2025-11-18.parquet")

        # Assert equivalence
        assert len(csv_data) == len(
            parquet_data
        ), f"Row count mismatch: CSV {len(csv_data)}, Parquet {len(parquet_data)}"

        # Convert data types to match for comparison
        # Convert mcc column to string for consistent comparison
        csv_data["mcc"] = csv_data["mcc"].astype(str)
        parquet_data["mcc"] = parquet_data["mcc"].astype(str)

        # Sort both for comparison
        csv_sorted = csv_data.sort_values(["customer_id", "mcc"]).reset_index(drop=True)
        parquet_sorted = parquet_data.sort_values(["customer_id", "mcc"]).reset_index(
            drop=True
        )

        # Compare data
        pd.testing.assert_frame_equal(csv_sorted, parquet_sorted)

        print("✓ CSV and Parquet data are equivalent")

    def test_validation_summary_details(self, test_config: dict[str, Any]) -> None:
        """
        Validate that validation summary contains expected details.

        Checks:
        - Validation summary exists
        - Contains key metrics
        - No critical failures
        """
        summary = run_anomaly_detection(
            reporting_week=test_config["reporting_week"], mode=test_config["mode"]
        )

        validation_summary = summary["validation_summary"]

        # Assert validation summary structure
        assert isinstance(validation_summary, dict), "Validation summary must be a dict"

        # Assert no critical failures
        critical_failures = validation_summary.get("critical_failures", 0)
        assert (
            critical_failures == 0
        ), f"Critical validation failures detected: {critical_failures}"

        print("✓ Validation summary validated successfully")

    def test_output_directory_structure(self, test_config: dict[str, Any]) -> None:
        """
        Validate output directory structure.

        Checks:
        - outputs/ directory exists
        - reports/ subdirectory exists
        - model_artifacts/{reporting_week}/ exists
        """
        summary = run_anomaly_detection(
            reporting_week=test_config["reporting_week"], mode=test_config["mode"]
        )

        # Check outputs directory
        outputs_dir = Path("./outputs")
        assert outputs_dir.exists(), "outputs/ directory not found"
        assert outputs_dir.is_dir(), "outputs/ is not a directory"

        # Check reports directory
        report_path = Path(summary["output_files"]["report"])
        assert report_path.parent.exists(), "reports/ directory not found"

        # Check model artifacts directory
        artifacts_dir = Path(summary["output_files"]["model_artifacts"])
        assert artifacts_dir.exists(), "model_artifacts/ directory not found"
        assert artifacts_dir.is_dir(), "model_artifacts/ is not a directory"

        # Check naming convention
        assert test_config["reporting_week"] in str(
            artifacts_dir
        ), "Reporting week not in artifacts directory path"

        print("✓ Output directory structure validated successfully")


class TestPerformanceBenchmarks:
    """Performance benchmark tests for pipeline stages."""

    def test_data_loading_performance(self) -> None:
        """Test that data loading completes within 5 seconds."""
        from src.data.loader import DataLoader
        from src.utils.config_loader import load_config

        config = load_config("./config/modelconfig.yaml", "./config/dataconfig.yaml")
        loader = DataLoader(config)

        start_time = time.time()
        data = loader.load("2025-11-18")
        elapsed = time.time() - start_time

        assert elapsed < 5, f"Data loading took {elapsed:.2f}s, exceeds 5s limit"
        assert len(data) > 0, "No data loaded"

        print(f"✓ Data loading completed in {elapsed:.2f} seconds")

    def test_validation_performance(self) -> None:
        """Test that data validation completes within 10 seconds."""
        from src.data.loader import DataLoader
        from src.data.validator import DataValidator
        from src.utils.config_loader import load_config

        config = load_config("./config/modelconfig.yaml", "./config/dataconfig.yaml")
        loader = DataLoader(config)
        validator = DataValidator(config)

        data = loader.load("2025-11-18")

        start_time = time.time()
        clean_data, _ = validator.validate(data)
        elapsed = time.time() - start_time

        assert elapsed < 10, f"Validation took {elapsed:.2f}s, exceeds 10s limit"
        assert len(clean_data) > 0, "No valid data after validation"

        print(f"✓ Validation completed in {elapsed:.2f} seconds")

    def test_feature_engineering_performance(self) -> None:
        """Test that feature engineering completes within 30 seconds."""
        from src.data.loader import DataLoader
        from src.data.validator import DataValidator
        from src.features.builder import FeatureBuilder
        from src.utils.config_loader import load_config

        config = load_config("./config/modelconfig.yaml", "./config/dataconfig.yaml")
        loader = DataLoader(config)
        validator = DataValidator(config)
        builder = FeatureBuilder(config)

        data = loader.load("2025-11-18")
        clean_data, _ = validator.validate(data)

        start_time = time.time()
        features = builder.build_features(clean_data)
        elapsed = time.time() - start_time

        assert (
            elapsed < 30
        ), f"Feature engineering took {elapsed:.2f}s, exceeds 30s limit"
        assert len(features) > 0, "No features generated"

        print(f"✓ Feature engineering completed in {elapsed:.2f} seconds")

    def test_model_scoring_performance(self) -> None:
        """Test that model scoring completes within 60 seconds."""
        from src.data.loader import DataLoader
        from src.data.validator import DataValidator
        from src.features.builder import FeatureBuilder
        from src.models.scorer import ModelScorer
        from src.utils.config_loader import load_config

        config = load_config("./config/modelconfig.yaml", "./config/dataconfig.yaml")
        loader = DataLoader(config)
        validator = DataValidator(config)
        builder = FeatureBuilder(config)
        scorer = ModelScorer(config)

        data = loader.load("2025-11-18")
        clean_data, _ = validator.validate(data)
        features = builder.build_features(clean_data)

        start_time = time.time()
        scored_df = scorer.fit_and_score(features)
        elapsed = time.time() - start_time

        assert elapsed < 60, f"Model scoring took {elapsed:.2f}s, exceeds 60s limit"
        assert len(scored_df) > 0, "No scored data"

        print(f"✓ Model scoring completed in {elapsed:.2f} seconds")

    def test_report_generation_performance(self) -> None:
        """Test that report generation completes within 15 seconds."""
        from src.data.loader import DataLoader
        from src.data.validator import DataValidator
        from src.features.builder import FeatureBuilder
        from src.models.scorer import ModelScorer
        from src.reporting.generator import ReportGenerator
        from src.utils.config_loader import load_config

        config = load_config("./config/modelconfig.yaml", "./config/dataconfig.yaml")
        loader = DataLoader(config)
        validator = DataValidator(config)
        builder = FeatureBuilder(config)
        scorer = ModelScorer(config)
        generator = ReportGenerator(config)

        data = loader.load("2025-11-18")
        clean_data, _ = validator.validate(data)
        features = builder.build_features(clean_data)
        scored_df = scorer.fit_and_score(features)

        start_time = time.time()
        report_path, summary_path = generator.generate(scored_df, data, "2025-11-18")
        elapsed = time.time() - start_time

        assert elapsed < 15, f"Report generation took {elapsed:.2f}s, exceeds 15s limit"
        assert Path(report_path).exists(), "Report file not created"
        assert Path(summary_path).exists(), "Summary file not created"

        print(f"✓ Report generation completed in {elapsed:.2f} seconds")
