"""
Generate sample data for E2E testing.

This script creates realistic sample transaction data with:
- Multiple reporting weeks
- Diverse customer spending patterns
- Edge cases for anomaly detection testing
- Both CSV and Parquet output formats
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# MCC codes with descriptions
MCC_CODES = {
    "5411": "Grocery Stores",
    "5812": "Restaurants",
    "5541": "Gas Stations",
    "5999": "Misc Retail",
    "5732": "Electronics",
    "5912": "Drug Stores",
    "5311": "Department Stores",
    "5814": "Fast Food",
}


def generate_normal_customers(
    n_customers: int, reporting_week: str, seed: int = 42
) -> list[dict[str, Any]]:
    """
    Generate normal customer transaction data.

    Args:
        n_customers: Number of customers to generate
        reporting_week: ISO week start date
        seed: Random seed for reproducibility

    Returns:
        List of transaction dictionaries
    """
    np.random.seed(seed)
    data = []

    for i in range(1, n_customers + 1):
        customer_id = f"CUST{i:05d}"

        # Random number of active MCCs per customer (1-6)
        n_mccs = np.random.randint(1, 7)
        active_mccs = np.random.choice(list(MCC_CODES.keys()), n_mccs, replace=False)

        for mcc in active_mccs:
            # Generate realistic spend and transaction patterns
            if np.random.random() < 0.7:  # 70% low spenders
                txn_count = np.random.randint(1, 20)
                spend = np.random.uniform(50, 1000)
            elif np.random.random() < 0.85:  # 15% medium spenders
                txn_count = np.random.randint(10, 50)
                spend = np.random.uniform(500, 3000)
            else:  # 15% high spenders
                txn_count = np.random.randint(20, 100)
                spend = np.random.uniform(2000, 10000)

            avg_ticket = spend / txn_count

            data.append(
                {
                    "customer_id": customer_id,
                    "reporting_week": reporting_week,
                    "mcc": mcc,
                    "spend_amount": round(spend, 2),
                    "transaction_count": txn_count,
                    "avg_ticket_amount": round(avg_ticket, 2),
                }
            )

    return data


def generate_edge_cases(reporting_week: str) -> list[dict[str, Any]]:
    """
    Generate edge case customers for testing specific anomaly patterns.

    Args:
        reporting_week: ISO week start date

    Returns:
        List of edge case transaction dictionaries
    """
    edge_cases = []

    # Edge Case 1: High-growth customer (will show spend_growth_4w > 0.5)
    edge_cases.extend(
        [
            {
                "customer_id": "EDGE001",
                "reporting_week": reporting_week,
                "mcc": "5411",
                "spend_amount": 8000.0,
                "transaction_count": 80,
                "avg_ticket_amount": 100.0,
            },
            {
                "customer_id": "EDGE001",
                "reporting_week": reporting_week,
                "mcc": "5812",
                "spend_amount": 3000.0,
                "transaction_count": 50,
                "avg_ticket_amount": 60.0,
            },
        ]
    )

    # Edge Case 2: High-concentration customer (herfindahl > 0.7)
    edge_cases.append(
        {
            "customer_id": "EDGE002",
            "reporting_week": reporting_week,
            "mcc": "5812",
            "spend_amount": 9500.0,
            "transaction_count": 100,
            "avg_ticket_amount": 95.0,
        }
    )
    edge_cases.append(
        {
            "customer_id": "EDGE002",
            "reporting_week": reporting_week,
            "mcc": "5411",
            "spend_amount": 500.0,
            "transaction_count": 10,
            "avg_ticket_amount": 50.0,
        }
    )

    # Edge Case 3: Unusual ticket size (avg_ticket >> median)
    edge_cases.append(
        {
            "customer_id": "EDGE003",
            "reporting_week": reporting_week,
            "mcc": "5732",
            "spend_amount": 15000.0,
            "transaction_count": 5,
            "avg_ticket_amount": 3000.0,
        }
    )

    # Edge Case 4: Single MCC customer
    edge_cases.append(
        {
            "customer_id": "EDGE004",
            "reporting_week": reporting_week,
            "mcc": "5411",
            "spend_amount": 2000.0,
            "transaction_count": 40,
            "avg_ticket_amount": 50.0,
        }
    )

    # Edge Case 5: New high spender
    edge_cases.extend(
        [
            {
                "customer_id": "EDGE005",
                "reporting_week": reporting_week,
                "mcc": "5732",
                "spend_amount": 12000.0,
                "transaction_count": 30,
                "avg_ticket_amount": 400.0,
            },
            {
                "customer_id": "EDGE005",
                "reporting_week": reporting_week,
                "mcc": "5999",
                "spend_amount": 8000.0,
                "transaction_count": 20,
                "avg_ticket_amount": 400.0,
            },
        ]
    )

    # Edge Case 6: Zero spend (boundary case)
    edge_cases.append(
        {
            "customer_id": "EDGE006",
            "reporting_week": reporting_week,
            "mcc": "5411",
            "spend_amount": 0.0,
            "transaction_count": 0,
            "avg_ticket_amount": 0.0,
        }
    )

    # Edge Case 7: Many small transactions
    edge_cases.append(
        {
            "customer_id": "EDGE007",
            "reporting_week": reporting_week,
            "mcc": "5814",
            "spend_amount": 500.0,
            "transaction_count": 200,
            "avg_ticket_amount": 2.5,
        }
    )

    return edge_cases


def generate_historical_data(
    customer_ids: list[str], reporting_week: str, seed: int = 42
) -> list[dict[str, Any]]:
    """
    Generate historical data for delta features (12-week lookback).

    Args:
        customer_ids: List of customer IDs to generate history for
        reporting_week: Current reporting week
        seed: Random seed for reproducibility

    Returns:
        List of historical transaction dictionaries
    """
    np.random.seed(seed + 1000)  # Different seed for historical data
    data = []

    # Parse reporting week
    current_date = datetime.strptime(reporting_week, "%Y-%m-%d")

    # Generate 12 weeks of history
    for week_offset in range(1, 13):
        historical_week = (current_date - timedelta(weeks=week_offset)).strftime(
            "%Y-%m-%d"
        )

        # Sample 80% of customers for each historical week
        sampled_customers = np.random.choice(
            customer_ids, size=int(len(customer_ids) * 0.8), replace=False
        )

        for customer_id in sampled_customers:
            # Random number of MCCs (fewer in history)
            n_mccs = np.random.randint(1, 5)
            active_mccs = np.random.choice(
                list(MCC_CODES.keys()), n_mccs, replace=False
            )

            for mcc in active_mccs:
                # Historical spend patterns (slightly lower than current)
                txn_count = np.random.randint(1, 30)
                spend = np.random.uniform(100, 2000)
                avg_ticket = spend / txn_count

                data.append(
                    {
                        "customer_id": customer_id,
                        "reporting_week": historical_week,
                        "mcc": mcc,
                        "spend_amount": round(spend, 2),
                        "transaction_count": txn_count,
                        "avg_ticket_amount": round(avg_ticket, 2),
                    }
                )

    return data


def save_data(
    data: list[dict[str, Any]], reporting_week: str, output_dir: Path
) -> None:
    """
    Save data in both CSV and Parquet formats.

    Args:
        data: List of transaction dictionaries
        reporting_week: ISO week start date
        output_dir: Output directory path
    """
    df = pd.DataFrame(data)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_path = output_dir / f"transactions_{reporting_week}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(
        f"Saved {len(df)} rows to CSV: {csv_path} "
        f"({csv_path.stat().st_size / 1024:.2f} KB)"
    )

    # Save as Parquet
    parquet_path = output_dir / f"transactions_{reporting_week}.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(
        f"Saved {len(df)} rows to Parquet: {parquet_path} "
        f"({parquet_path.stat().st_size / 1024:.2f} KB)"
    )


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate sample data for E2E testing")
    parser.add_argument(
        "--n-customers",
        type=int,
        default=500,
        help="Number of normal customers to generate",
    )
    parser.add_argument(
        "--reporting-week",
        type=str,
        default="2025-11-18",
        help="Current reporting week (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasamples",
        help="Output directory for data files",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--with-historical",
        action="store_true",
        help="Generate 12 weeks of historical data",
    )

    args = parser.parse_args()

    logger.info("Starting sample data generation")
    logger.info(f"Parameters: {vars(args)}")

    output_dir = Path(args.output_dir)

    # Generate current week data
    logger.info(f"Generating {args.n_customers} normal customers")
    normal_data = generate_normal_customers(
        args.n_customers, args.reporting_week, args.seed
    )

    logger.info("Generating edge case customers")
    edge_data = generate_edge_cases(args.reporting_week)

    # Combine data
    all_data = normal_data + edge_data
    logger.info(f"Total transactions for {args.reporting_week}: {len(all_data)}")

    # Save current week data
    save_data(all_data, args.reporting_week, output_dir)

    # Generate historical data if requested
    if args.with_historical:
        logger.info("Generating historical data (12 weeks)")

        # Get all customer IDs
        customer_ids = list(
            set(
                [record["customer_id"] for record in all_data]
                + [f"CUST{i:05d}" for i in range(1, args.n_customers + 1)]
            )
        )

        historical_data = generate_historical_data(
            customer_ids, args.reporting_week, args.seed
        )
        logger.info(f"Generated {len(historical_data)} historical transactions")

        # Save historical data as a single file
        historical_week = "historical"
        save_data(historical_data, historical_week, output_dir)

    logger.info("Sample data generation complete!")


if __name__ == "__main__":
    main()
