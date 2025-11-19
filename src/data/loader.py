"""Data loader module for loading transactional data from various sources.

This module provides the DataLoader class which supports loading data from
SQL databases and CSV/Parquet files based on configuration.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.exceptions import ConfigurationError, DataLoadError

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate raw transactional data from configured sources.

    The DataLoader dispatches to SQL or CSV loading based on configuration
    and returns a pandas DataFrame with the expected schema.

    Attributes:
        config: Configuration dictionary from dataconfig.yaml
        source_type: Type of data source ('sql' or 'csv')
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with dataconfig.yaml settings.

        Args:
            config: Configuration dictionary containing datasource settings

        Raises:
            ConfigurationError: If required configuration is missing
        """
        if "datasource" not in config:
            raise ConfigurationError("Missing 'datasource' in configuration")

        self.config = config
        self.source_type = config["datasource"].get("type")

        if self.source_type not in ["sql", "csv"]:
            raise ConfigurationError(
                f"Invalid datasource type: {self.source_type}. Must be 'sql' or 'csv'"
            )

        logger.info(f"DataLoader initialized with source type: {self.source_type}")

    def load(self, reporting_week: str) -> pd.DataFrame:
        """Load data for specified reporting week.

        This method dispatches to the appropriate loader (SQL or CSV) based
        on the configured source type.

        Args:
            reporting_week: ISO week start date (e.g., '2025-11-18')

        Returns:
            DataFrame with columns: customer_id, reporting_week, mcc,
                                   spend_amount, transaction_count, avg_ticket_amount

        Raises:
            DataLoadError: If loading fails from either source
        """
        logger.info(f"Loading data for week {reporting_week}")
        start_time = time.time()

        try:
            if self.source_type == "sql":
                df = self.load_from_sql(reporting_week)
            else:
                df = self.load_from_csv(reporting_week)

            elapsed = time.time() - start_time
            logger.info(
                f"Loaded {len(df)} rows from {self.source_type} in {elapsed:.2f} seconds"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to load data for week {reporting_week}: {str(e)}")
            raise DataLoadError(f"Data load failed: {str(e)}") from e

    def load_from_sql(self, reporting_week: str) -> pd.DataFrame:
        """Execute SQL query via configured connector.

        Implements retry logic with exponential backoff and query timeout.

        Args:
            reporting_week: ISO week start date for query substitution

        Returns:
            DataFrame with query results

        Raises:
            DataLoadError: If SQL execution fails after all retries
        """
        sql_config = self.config["datasource"].get("sql", {})
        connection_string = sql_config.get("connection_string")
        query_template = sql_config.get("query_template")
        timeout = sql_config.get("timeout", 60)
        max_retries = sql_config.get("max_retries", 3)

        if not connection_string:
            # Try to get from environment variable
            connection_string = os.getenv("DB_CONNECTION_STRING")
            if not connection_string:
                raise ConfigurationError(
                    "Missing SQL connection_string in config and DB_CONNECTION_STRING env var"
                )

        if not query_template:
            raise ConfigurationError("Missing SQL query_template in configuration")

        # Substitute reporting_week in query
        query = query_template.format(reporting_week=reporting_week)

        # Retry logic with exponential backoff
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Executing SQL query (attempt {attempt}/{max_retries})")

                # Import here to avoid dependency issues if sqlalchemy not installed
                try:
                    from sqlalchemy import create_engine
                except ImportError as e:
                    raise DataLoadError(
                        "sqlalchemy is required for SQL data sources. "
                        "Install with: pip install sqlalchemy"
                    ) from e

                # Create engine with connection timeout
                engine = create_engine(
                    connection_string,
                    connect_args={"connect_timeout": timeout},
                )

                # Execute query with timeout using pandas
                # Note: pandas read_sql doesn't directly support query timeout,
                # but the connection timeout will prevent indefinite hangs
                df = pd.read_sql_query(query, engine, params=None)

                logger.info(
                    f"SQL query executed successfully: {len(df)} rows retrieved"
                )
                return df

            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2**attempt  # Exponential backoff: 2, 4, 8 seconds
                    logger.warning(
                        f"SQL query failed (attempt {attempt}/{max_retries}): {str(e)}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"SQL query failed after {max_retries} attempts: {str(e)}"
                    )
                    raise DataLoadError(
                        f"SQL query failed after {max_retries} retries: {str(e)}"
                    ) from e

        # This should never be reached, but for type safety
        raise DataLoadError("Unexpected error in SQL loading")

    def load_from_csv(self, reporting_week: str) -> pd.DataFrame:
        """Load from CSV/Parquet file.

        Supports both CSV and Parquet formats with automatic detection by extension.

        Args:
            reporting_week: ISO week start date for filename substitution

        Returns:
            DataFrame with file contents

        Raises:
            DataLoadError: If file not found or loading fails
        """
        csv_config = self.config["datasource"].get("csv", {})
        directory = csv_config.get("directory", "./datasamples")
        filename_pattern = csv_config.get(
            "filename_pattern", "transactions_{reporting_week}.csv"
        )

        # Substitute reporting_week in filename
        filename = filename_pattern.format(reporting_week=reporting_week)
        filepath = Path(directory) / filename

        if not filepath.exists():
            raise DataLoadError(
                f"File not found: {filepath}. "
                f"Available files: {list(Path(directory).glob('*'))}"
            )

        try:
            # Detect format by extension
            if filepath.suffix.lower() == ".parquet":
                logger.info(f"Loading Parquet file: {filepath}")
                df = pd.read_parquet(filepath)
            elif filepath.suffix.lower() == ".csv":
                logger.info(f"Loading CSV file: {filepath}")
                df = pd.read_csv(filepath)
            else:
                raise DataLoadError(
                    f"Unsupported file format: {filepath.suffix}. "
                    "Must be .csv or .parquet"
                )

            file_size = filepath.stat().st_size / (1024 * 1024)  # Size in MB
            logger.info(
                f"Loaded {len(df)} rows from {filepath.name} ({file_size:.2f} MB)"
            )
            return df

        except pd.errors.EmptyDataError as e:
            raise DataLoadError(f"File is empty: {filepath}") from e
        except Exception as e:
            logger.error(f"Failed to load file {filepath}: {str(e)}")
            raise DataLoadError(f"File load failed: {str(e)}") from e
