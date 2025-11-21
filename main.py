"""
Main entry point for the anomaly detection system.

This module orchestrates the complete anomaly detection pipeline:
1. Load and validate data
2. Engineer features
3. Train/apply the anomaly detection model
4. Generate reports

Usage:
    python main.py --reporting-week 2025-11-18
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.features.builder import FeatureBuilder
from src.models.scorer import ModelScorer
from src.reporting.generator import ReportGenerator
from src.utils.config_loader import load_config
from src.utils.exceptions import (
    ConfigurationError,
    DataLoadError,
    DataValidationError,
)

logger = logging.getLogger(__name__)


def run_anomaly_detection(
    reporting_week: str,
    config_path: str = "./config/modelconfig.yaml",
    data_config_path: str = "./config/dataconfig.yaml",
    mode: str = "weekly",
) -> dict[str, Any]:
    """
    Main orchestration function for anomaly detection pipeline.

    Args:
        reporting_week: ISO date string (YYYY-MM-DD) for target week
        config_path: Path to model configuration YAML
        data_config_path: Path to data configuration YAML
        mode: Execution mode ('weekly', 'monthly', 'adhoc')

    Returns:
        Execution summary dict with paths to outputs and metrics

    Raises:
        ConfigurationError: If configuration is invalid
        DataLoadError: If data loading fails
        DataValidationError: If critical validation failures occur
    """
    start_time = time.time()
    logger.info("Starting anomaly detection pipeline")
    logger.info(f"Reporting week: {reporting_week}, Mode: {mode}")

    try:
        # 1. Load config
        logger.info("Loading configuration files")
        config = load_config(config_path, data_config_path)
        logger.info("Configuration loaded successfully")

        # 2. Initialize components
        logger.info("Initializing pipeline components")
        loader = DataLoader(config)
        validator = DataValidator(config)
        builder = FeatureBuilder(config)
        scorer = ModelScorer(config)
        generator = ReportGenerator(config)
        logger.info("Components initialized successfully")

        # 3. Load data
        logger.info(f"Loading data for reporting week: {reporting_week}")
        raw_data = loader.load(reporting_week)
        data_loaded_rows = len(raw_data)
        logger.info(f"Loaded {data_loaded_rows} rows from data source")

        # 4. Validate data
        logger.info("Validating data quality")
        clean_data, validation_summary = validator.validate(raw_data)
        valid_rows = len(clean_data)
        logger.info(
            f"Data validation complete: {valid_rows} valid rows, "
            f"{data_loaded_rows - valid_rows} rejected"
        )

        # Check for critical validation failures
        if validation_summary.get("critical_failures", 0) > 0:
            raise DataValidationError(
                f"Critical validation failures detected: {validation_summary}"
            )

        # 5. Build features
        logger.info("Building features")
        features = builder.build_features(clean_data)
        customers_scored = len(features)
        feature_count = len(features.columns) - 1  # Exclude customer_id
        logger.info(f"Built {feature_count} features for {customers_scored} customers")

        # 6. Score anomalies
        logger.info("Training model and scoring customers")
        scored_df = scorer.fit_and_score(features)
        top_anomalies_count = (
            len(scored_df[scored_df["anomaly_label"] == 1])
            if "anomaly_label" in scored_df.columns
            else 0
        )
        logger.info(
            f"Scored {customers_scored} customers, "
            f"identified {top_anomalies_count} anomalies"
        )

        # 7. Generate reports
        logger.info("Generating reports")
        report_path, summary_path = generator.generate(
            scored_df, raw_data, reporting_week
        )
        logger.info(f"Reports generated: {report_path}, {summary_path}")

        # 8. Save artifacts
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)
        model_artifacts_dir = output_dir / "model_artifacts" / reporting_week
        model_artifacts_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model artifacts to {model_artifacts_dir}")
        scorer.save_artifacts(str(output_dir), reporting_week)
        logger.info("Model artifacts saved successfully")

        # Calculate execution time
        execution_time = time.time() - start_time

        logger.info(f"Pipeline completed successfully in {execution_time:.1f} seconds")

        return {
            "status": "success",
            "reporting_week": reporting_week,
            "execution_time_seconds": int(execution_time),
            "data_loaded_rows": data_loaded_rows,
            "valid_rows": valid_rows,
            "customers_scored": customers_scored,
            "top_anomalies_count": top_anomalies_count,
            "output_files": {
                "report": report_path,
                "summary": summary_path,
                "model_artifacts": str(model_artifacts_dir),
            },
            "validation_summary": validation_summary,
            "config_snapshot": {
                "model_config": config_path,
                "data_config": data_config_path,
                "mode": mode,
            },
        }

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return {
            "status": "failed",
            "error": f"Configuration error: {str(e)}",
            "error_type": "configuration",
        }

    except DataLoadError as e:
        logger.error(f"Data load error: {e}")
        return {
            "status": "failed",
            "error": f"Data load error: {str(e)}",
            "error_type": "data_load",
        }

    except DataValidationError as e:
        logger.error(f"Data validation error: {e}")
        return {
            "status": "failed",
            "error": f"Data validation error: {str(e)}",
            "error_type": "data_validation",
        }

    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}", exc_info=True)
        return {
            "status": "failed",
            "error": f"Unexpected error: {str(e)}",
            "error_type": "unexpected",
        }


if __name__ == "__main__":
    # Simple CLI for direct execution
    import argparse

    parser = argparse.ArgumentParser(
        description="Card Portfolio Anomaly Detection System"
    )
    parser.add_argument(
        "--reporting-week",
        required=True,
        help="Reporting week in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--model-config",
        default="config/modelconfig.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--data-config",
        default="config/dataconfig.yaml",
        help="Path to data configuration file",
    )
    parser.add_argument(
        "--mode",
        default="weekly",
        choices=["weekly", "monthly", "adhoc"],
        help="Execution mode",
    )

    args = parser.parse_args()

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result = run_anomaly_detection(
        reporting_week=args.reporting_week,
        config_path=args.model_config,
        data_config_path=args.data_config,
        mode=args.mode,
    )

    # Print result summary
    if result["status"] == "success":
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUCCESSFUL")
        print("=" * 60)
        print(f"Reporting Week: {result['reporting_week']}")
        print(f"Execution Time: {result['execution_time_seconds']}s")
        print(f"Data Loaded: {result['data_loaded_rows']} rows")
        print(f"Valid Rows: {result['valid_rows']}")
        print(f"Customers Scored: {result['customers_scored']}")
        print(f"Anomalies Detected: {result['top_anomalies_count']}")
        print("\nOutput Files:")
        for key, path in result["output_files"].items():
            print(f"  {key}: {path}")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION FAILED")
        print("=" * 60)
        print(f"Error Type: {result.get('error_type', 'unknown')}")
        print(f"Error: {result['error']}")
        print("=" * 60)
        sys.exit(1)
