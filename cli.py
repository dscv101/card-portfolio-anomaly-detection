"""
Command-line interface for the anomaly detection system.

Provides a user-friendly CLI for running various operations:
- detect: Run full anomaly detection pipeline
- validate: Validate data without running full pipeline
- backtest: Run pipeline over a date range

Usage:
    python cli.py detect --week 2025-11-18 --mode weekly
    python cli.py validate --data-file ./datasamples/sample_week_current.parquet
    python cli.py backtest --start-week 2025-09-01 --end-week 2025-11-18
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from main import run_anomaly_detection
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def detect_command(args: argparse.Namespace) -> int:
    """
    Run anomaly detection pipeline for a specific week.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error, 2 = config error,
                   3 = validation failure, 4 = model failure)
    """
    print(f"Running anomaly detection for week: {args.week}")
    print(f"Mode: {args.mode}")
    print(f"Model config: {args.model_config}")
    print(f"Data config: {args.data_config}")
    print("-" * 60)

    try:
        result = run_anomaly_detection(
            reporting_week=args.week,
            config_path=args.model_config,
            data_config_path=args.data_config,
            mode=args.mode,
        )

        if result["status"] == "success":
            print("\n✅ SUCCESS")
            print(f"   Execution time: {result['execution_time_seconds']}s")
            print(f"   Data loaded: {result['data_loaded_rows']} rows")
            print(f"   Valid rows: {result['valid_rows']}")
            print(f"   Customers scored: {result['customers_scored']}")
            print(f"   Anomalies detected: {result['top_anomalies_count']}")
            print("\nOutput files:")
            for key, path in result["output_files"].items():
                print(f"   {key}: {path}")
            return 0
        else:
            print(f"\n❌ FAILED: {result['error']}")
            error_type = result.get("error_type", "unexpected")
            exit_code_map = {
                "configuration": 2,
                "data_validation": 3,
                "model": 4,
                "data_load": 1,
                "unexpected": 1,
            }
            return exit_code_map.get(error_type, 1)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"CLI error: {e}", exc_info=True)
        return 1


def validate_command(args: argparse.Namespace) -> int:
    """
    Validate data file without running full pipeline.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = valid, 3 = validation failures)
    """
    print(f"Validating data file: {args.data_file}")
    print("-" * 60)

    try:
        # Load configurations
        config = load_config(args.model_config, args.data_config)

        # Load data file
        data_file = Path(args.data_file)
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            return 1

        if data_file.suffix == ".parquet":
            df = pd.read_parquet(data_file)
        elif data_file.suffix == ".csv":
            df = pd.read_csv(data_file)
        else:
            print(f"❌ Unsupported file format: {data_file.suffix}")
            return 1

        print(f"✓ Loaded {len(df)} rows from {data_file}")

        # Validate data
        validator = DataValidator(config)
        clean_data, validation_summary = validator.validate(df)

        print(f"\n📊 Validation Results:")
        print(f"   Total rows: {len(df)}")
        print(f"   Valid rows: {len(clean_data)}")
        print(f"   Rejected rows: {len(df) - len(clean_data)}")

        if validation_summary:
            print(f"\n   Critical failures: {validation_summary.get('critical_failures', 0)}")
            print(f"   Errors: {validation_summary.get('errors', 0)}")
            print(f"   Warnings: {validation_summary.get('warnings', 0)}")

            if validation_summary.get("critical_failures", 0) > 0:
                print(f"\n❌ VALIDATION FAILED: Critical issues detected")
                return 3

        print(f"\n✅ VALIDATION PASSED")
        return 0

    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        logger.error(f"Validation error: {e}", exc_info=True)
        return 1


def backtest_command(args: argparse.Namespace) -> int:
    """
    Run anomaly detection over a date range for backtesting.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = all successful, 1 = any failures)
    """
    print(f"Running backtest from {args.start_week} to {args.end_week}")
    print(f"Model config: {args.model_config}")
    print(f"Data config: {args.data_config}")
    print("-" * 60)

    try:
        # Parse dates
        start_date = datetime.strptime(args.start_week, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_week, "%Y-%m-%d")

        if start_date > end_date:
            print("❌ Start week must be before end week")
            return 1

        # Generate weekly dates
        current_date = start_date
        weeks: list[str] = []
        while current_date <= end_date:
            weeks.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=7)

        print(f"\nProcessing {len(weeks)} weeks...\n")

        # Track results
        results: list[dict[str, Any]] = []
        successes = 0
        failures = 0

        for week in weeks:
            print(f"Processing week: {week}")
            result = run_anomaly_detection(
                reporting_week=week,
                config_path=args.model_config,
                data_config_path=args.data_config,
                mode="backtest",
            )

            results.append(
                {
                    "week": week,
                    "status": result["status"],
                    "execution_time": result.get("execution_time_seconds", 0),
                    "anomalies": result.get("top_anomalies_count", 0),
                }
            )

            if result["status"] == "success":
                print(f"  ✅ Success - {result.get('top_anomalies_count', 0)} anomalies")
                successes += 1
            else:
                print(f"  ❌ Failed - {result.get('error', 'Unknown error')}")
                failures += 1
            print()

        # Summary
        print("=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        print(f"Total weeks processed: {len(weeks)}")
        print(f"Successful: {successes}")
        print(f"Failed: {failures}")

        # Save results to file
        output_file = Path("outputs") / "backtest_results.json"
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")

        return 0 if failures == 0 else 1

    except Exception as e:
        print(f"\n❌ Backtest error: {e}")
        logger.error(f"Backtest error: {e}", exc_info=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command-line arguments (for testing)

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Card Portfolio Anomaly Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run detection for a specific week
  python cli.py detect --week 2025-11-18

  # Validate a data file
  python cli.py validate --data-file ./datasamples/sample_week_current.parquet

  # Backtest over a date range
  python cli.py backtest --start-week 2025-09-01 --end-week 2025-11-18
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Detect command
    detect_parser = subparsers.add_parser(
        "detect", help="Run anomaly detection for a specific week"
    )
    detect_parser.add_argument(
        "--week", required=True, help="Reporting week in YYYY-MM-DD format"
    )
    detect_parser.add_argument(
        "--mode",
        default="weekly",
        choices=["weekly", "monthly", "adhoc"],
        help="Execution mode (default: weekly)",
    )
    detect_parser.add_argument(
        "--model-config",
        default="config/modelconfig.yaml",
        help="Path to model configuration file",
    )
    detect_parser.add_argument(
        "--data-config",
        default="config/dataconfig.yaml",
        help="Path to data configuration file",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate data file without running full pipeline"
    )
    validate_parser.add_argument(
        "--data-file", required=True, help="Path to data file (.parquet or .csv)"
    )
    validate_parser.add_argument(
        "--model-config",
        default="config/modelconfig.yaml",
        help="Path to model configuration file",
    )
    validate_parser.add_argument(
        "--data-config",
        default="config/dataconfig.yaml",
        help="Path to data configuration file",
    )

    # Backtest command
    backtest_parser = subparsers.add_parser(
        "backtest", help="Run anomaly detection over a date range"
    )
    backtest_parser.add_argument(
        "--start-week", required=True, help="Start week in YYYY-MM-DD format"
    )
    backtest_parser.add_argument(
        "--end-week", required=True, help="End week in YYYY-MM-DD format"
    )
    backtest_parser.add_argument(
        "--model-config",
        default="config/modelconfig.yaml",
        help="Path to model configuration file",
    )
    backtest_parser.add_argument(
        "--data-config",
        default="config/dataconfig.yaml",
        help="Path to data configuration file",
    )

    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate command handler
    if args.command == "detect":
        return detect_command(args)
    elif args.command == "validate":
        return validate_command(args)
    elif args.command == "backtest":
        return backtest_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
