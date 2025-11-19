"""
Command-line interface for the anomaly detection system.

Provides a user-friendly CLI for running various operations:
- Data validation
- Feature engineering
- Model training
- Anomaly detection
- Report generation

Usage:
    python cli.py validate --data-config config/dataconfig.yaml
    python cli.py detect --model-config config/modelconfig.yaml
"""

import argparse
import sys
from pathlib import Path


def validate_command(args: argparse.Namespace) -> int:
    """
    Validate data sources and configuration.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    print(f"Validating data with config: {args.data_config}")
    print("Implementation coming in Phase 1...")
    return 0


def detect_command(args: argparse.Namespace) -> int:
    """
    Run anomaly detection pipeline.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    print(f"Running detection with config: {args.model_config}")
    print("Implementation coming in Phase 2-3...")
    return 0


def report_command(args: argparse.Namespace) -> int:
    """
    Generate anomaly reports.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    print(f"Generating reports from: {args.input}")
    print("Implementation coming in Phase 4...")
    return 0


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
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate data and configuration"
    )
    validate_parser.add_argument(
        "--data-config",
        type=Path,
        default="config/dataconfig.yaml",
        help="Path to data configuration file",
    )

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Run anomaly detection")
    detect_parser.add_argument(
        "--model-config",
        type=Path,
        default="config/modelconfig.yaml",
        help="Path to model configuration file",
    )
    detect_parser.add_argument(
        "--data-config",
        type=Path,
        default="config/dataconfig.yaml",
        help="Path to data configuration file",
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument(
        "--input", type=Path, required=True, help="Input anomaly detection results"
    )
    report_parser.add_argument(
        "--output-dir", type=Path, default="outputs", help="Output directory"
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate command handler
    if args.command == "validate":
        return validate_command(args)
    elif args.command == "detect":
        return detect_command(args)
    elif args.command == "report":
        return report_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
