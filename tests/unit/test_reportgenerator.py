"""Unit tests for ReportGenerator class.

Tests cover:
- Initialization and configuration validation
- Input DataFrame validation
- Report generation orchestration
- Error handling for missing configuration
- Error handling for invalid inputs
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting.generator import ReportGenerator
from src.utils.exceptions import ReportGenerationError


@pytest.fixture
def valid_config():
    """Valid ReportGenerator configuration."""
    return {
        "reporting": {
            "topnanomalies": 20,
            "rules": {
                "minspend": 100.0,
                "mintransactions": 5,
            },
        }
    }


@pytest.fixture
def sample_scored_df():
    """Sample scored DataFrame with anomaly scores."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "customer_id": [f"CUST{i:04d}" for i in range(50)],
            "reporting_week": ["2025-11-18"] * 50,
            "anomaly_score": np.random.uniform(-0.5, 0.5, 50),
            "anomaly_label": np.random.choice([-1, 1], 50),
            "total_spend": np.random.uniform(100, 10000, 50),
            "total_transactions": np.random.randint(5, 100, 50),
        }
    )


@pytest.fixture
def sample_raw_df():
    """Sample raw transaction DataFrame."""
    np.random.seed(42)
    customers = [f"CUST{i:04d}" for i in range(50)]
    mccs = ["5411", "5812", "5999", "4511", "7011"]

    # Generate transaction data for each customer across multiple MCCs
    data = []
    for customer in customers:
        for mcc in np.random.choice(mccs, size=3, replace=False):
            data.append(
                {
                    "customer_id": customer,
                    "mcc": mcc,
                    "spend_amount": np.random.uniform(50, 2000),
                    "transaction_count": np.random.randint(1, 20),
                }
            )

    return pd.DataFrame(data)


class TestReportGeneratorInitialization:
    """Test ReportGenerator initialization and configuration validation."""

    def test_initialization_with_valid_config(self, valid_config):
        """Test successful initialization with valid configuration."""
        generator = ReportGenerator(valid_config)

        assert generator.config == valid_config["reporting"]
        assert generator.logger is not None

    def test_initialization_missing_reporting_key(self):
        """Test initialization fails without 'reporting' key."""
        config = {"features": {}}

        with pytest.raises(
            ReportGenerationError, match="Missing 'reporting' in configuration"
        ):
            ReportGenerator(config)

    def test_initialization_missing_topnanomalies(self):
        """Test initialization fails without 'topnanomalies' key."""
        config = {"reporting": {"rules": {}}}

        with pytest.raises(
            ReportGenerationError,
            match="Missing 'topnanomalies' in reporting configuration",
        ):
            ReportGenerator(config)

    def test_initialization_logs_config(self, valid_config, caplog):
        """Test initialization logs configuration parameters."""
        with caplog.at_level(logging.INFO):
            ReportGenerator(valid_config)

        assert "ReportGenerator initialized with top_n=20" in caplog.text


class TestReportGeneratorValidation:
    """Test input DataFrame validation."""

    def test_validate_inputs_success(
        self, valid_config, sample_scored_df, sample_raw_df
    ):
        """Test successful validation with valid inputs."""
        generator = ReportGenerator(valid_config)

        # Should not raise any exception
        generator._validate_inputs(sample_scored_df, sample_raw_df)

    def test_validate_scored_df_missing_customer_id(
        self, valid_config, sample_scored_df, sample_raw_df
    ):
        """Test validation fails when scored_df missing customer_id."""
        generator = ReportGenerator(valid_config)
        scored_df_invalid = sample_scored_df.drop(columns=["customer_id"])

        with pytest.raises(
            ReportGenerationError,
            match="scored_df missing required columns.*customer_id",
        ):
            generator._validate_inputs(scored_df_invalid, sample_raw_df)

    def test_validate_scored_df_missing_anomaly_score(
        self, valid_config, sample_scored_df, sample_raw_df
    ):
        """Test validation fails when scored_df missing anomaly_score."""
        generator = ReportGenerator(valid_config)
        scored_df_invalid = sample_scored_df.drop(columns=["anomaly_score"])

        with pytest.raises(
            ReportGenerationError,
            match="scored_df missing required columns.*anomaly_score",
        ):
            generator._validate_inputs(scored_df_invalid, sample_raw_df)

    def test_validate_raw_df_missing_mcc(
        self, valid_config, sample_scored_df, sample_raw_df
    ):
        """Test validation fails when raw_df missing mcc."""
        generator = ReportGenerator(valid_config)
        raw_df_invalid = sample_raw_df.drop(columns=["mcc"])

        with pytest.raises(
            ReportGenerationError, match="raw_df missing required columns.*mcc"
        ):
            generator._validate_inputs(sample_scored_df, raw_df_invalid)

    def test_validate_empty_scored_df(
        self, valid_config, sample_scored_df, sample_raw_df
    ):
        """Test validation fails with empty scored_df."""
        generator = ReportGenerator(valid_config)
        empty_scored_df = sample_scored_df.iloc[:0]

        with pytest.raises(ReportGenerationError, match="scored_df is empty"):
            generator._validate_inputs(empty_scored_df, sample_raw_df)

    def test_validate_empty_raw_df(self, valid_config, sample_scored_df, sample_raw_df):
        """Test validation fails with empty raw_df."""
        generator = ReportGenerator(valid_config)
        empty_raw_df = sample_raw_df.iloc[:0]

        with pytest.raises(ReportGenerationError, match="raw_df is empty"):
            generator._validate_inputs(sample_scored_df, empty_raw_df)


class TestReportGeneratorGenerate:
    """Test report generation orchestration."""

    def test_generate_creates_output_directory(
        self, valid_config, sample_scored_df, sample_raw_df, tmp_path, monkeypatch
    ):
        """Test generate() creates output directory if it doesn't exist."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        generator = ReportGenerator(valid_config)
        reporting_week = "2025-11-18"

        report_path, summary_path = generator.generate(
            sample_scored_df, sample_raw_df, reporting_week
        )

        # Verify output directory was created
        output_dir = tmp_path / "outputs"
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_generate_returns_correct_paths(
        self, valid_config, sample_scored_df, sample_raw_df, tmp_path, monkeypatch
    ):
        """Test generate() returns correct output file paths."""
        monkeypatch.chdir(tmp_path)

        generator = ReportGenerator(valid_config)
        reporting_week = "2025-11-18"

        report_path, summary_path = generator.generate(
            sample_scored_df, sample_raw_df, reporting_week
        )

        assert report_path == "outputs/anomaly_report_2025-11-18.csv"
        assert summary_path == "outputs/anomaly_summary_2025-11-18.json"

    def test_generate_logs_execution(
        self,
        valid_config,
        sample_scored_df,
        sample_raw_df,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        """Test generate() logs execution steps."""
        monkeypatch.chdir(tmp_path)

        with caplog.at_level(logging.INFO):
            generator = ReportGenerator(valid_config)
            reporting_week = "2025-11-18"

            generator.generate(sample_scored_df, sample_raw_df, reporting_week)

        assert "Starting report generation for reporting_week=2025-11-18" in caplog.text
        assert "Report generation completed for 2025-11-18" in caplog.text
        assert "CSV report:" in caplog.text
        assert "JSON summary:" in caplog.text

    def test_generate_with_invalid_inputs_raises_error(
        self, valid_config, sample_scored_df, sample_raw_df, tmp_path, monkeypatch
    ):
        """Test generate() raises ReportGenerationError with invalid inputs."""
        monkeypatch.chdir(tmp_path)

        generator = ReportGenerator(valid_config)
        invalid_scored_df = sample_scored_df.drop(columns=["anomaly_score"])

        with pytest.raises(ReportGenerationError, match="Failed to generate report"):
            generator.generate(invalid_scored_df, sample_raw_df, "2025-11-18")


class TestReportGeneratorRankAnomalies:
    """Test rank_anomalies method."""

    def test_rank_anomalies_sorts_ascending(self, valid_config, sample_scored_df):
        """Test rank_anomalies() sorts by anomaly_score ascending."""
        generator = ReportGenerator(valid_config)

        ranked_df = generator.rank_anomalies(sample_scored_df)

        # Verify sorting - anomaly scores should be in ascending order
        assert (ranked_df["anomaly_score"].diff().dropna() >= 0).all()

    def test_rank_anomalies_preserves_all_rows(self, valid_config, sample_scored_df):
        """Test rank_anomalies() preserves all rows from input."""
        generator = ReportGenerator(valid_config)

        ranked_df = generator.rank_anomalies(sample_scored_df)

        assert len(ranked_df) == len(sample_scored_df)
        assert set(ranked_df.columns) == set(sample_scored_df.columns)

    def test_rank_anomalies_resets_index(self, valid_config, sample_scored_df):
        """Test rank_anomalies() resets index to start from 0."""
        generator = ReportGenerator(valid_config)

        ranked_df = generator.rank_anomalies(sample_scored_df)

        # Index should be sequential from 0
        assert ranked_df.index.tolist() == list(range(len(ranked_df)))

    def test_rank_anomalies_most_anomalous_first(self, valid_config):
        """Test rank_anomalies() places most anomalous (lowest score) first."""
        generator = ReportGenerator(valid_config)

        # Create DataFrame with known scores
        test_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3", "C4"],
                "anomaly_score": [0.5, -0.8, 0.2, -0.3],
                "anomaly_label": [1, -1, 1, -1],
            }
        )

        ranked_df = generator.rank_anomalies(test_df)

        # Most anomalous (lowest score) should be first
        assert ranked_df.iloc[0]["customer_id"] == "C2"
        assert ranked_df.iloc[0]["anomaly_score"] == -0.8
        assert ranked_df.iloc[-1]["customer_id"] == "C1"
        assert ranked_df.iloc[-1]["anomaly_score"] == 0.5

    def test_rank_anomalies_missing_anomaly_score(self, valid_config, sample_scored_df):
        """Test rank_anomalies() raises error when anomaly_score missing."""
        generator = ReportGenerator(valid_config)
        invalid_df = sample_scored_df.drop(columns=["anomaly_score"])

        with pytest.raises(
            ReportGenerationError,
            match="scored_df missing required column: anomaly_score",
        ):
            generator.rank_anomalies(invalid_df)

    def test_rank_anomalies_logs_execution(
        self, valid_config, sample_scored_df, caplog
    ):
        """Test rank_anomalies() logs execution details."""
        generator = ReportGenerator(valid_config)

        with caplog.at_level(logging.INFO):
            ranked_df = generator.rank_anomalies(sample_scored_df)

        assert f"Ranked {len(ranked_df)} customers by anomaly score" in caplog.text

    def test_rank_anomalies_handles_ties(self, valid_config):
        """Test rank_anomalies() handles tied anomaly scores."""
        generator = ReportGenerator(valid_config)

        # Create DataFrame with duplicate scores
        test_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "anomaly_score": [-0.5, -0.5, 0.3],
                "anomaly_label": [-1, -1, 1],
            }
        )

        ranked_df = generator.rank_anomalies(test_df)

        # Should handle ties gracefully
        assert len(ranked_df) == 3
        assert ranked_df.iloc[0]["anomaly_score"] == -0.5
        assert ranked_df.iloc[1]["anomaly_score"] == -0.5
        assert ranked_df.iloc[2]["anomaly_score"] == 0.3

    def test_rank_anomalies_with_single_row(self, valid_config):
        """Test rank_anomalies() works with single-row DataFrame."""
        generator = ReportGenerator(valid_config)

        test_df = pd.DataFrame(
            {"customer_id": ["C1"], "anomaly_score": [-0.5], "anomaly_label": [-1]}
        )

        ranked_df = generator.rank_anomalies(test_df)

        assert len(ranked_df) == 1
        assert ranked_df.iloc[0]["customer_id"] == "C1"


class TestReportGeneratorApplyTags:
    """Test apply_tags method."""

    def test_apply_tags_selects_top_n(self, valid_config, sample_scored_df):
        """Test apply_tags() selects exactly top N anomalies."""
        generator = ReportGenerator(valid_config)
        ranked_df = generator.rank_anomalies(sample_scored_df)

        top_df = generator.apply_tags(ranked_df, top_n=10)

        assert len(top_df) == 10

    def test_apply_tags_adds_required_columns(self, valid_config, sample_scored_df):
        """Test apply_tags() adds required tag columns."""
        generator = ReportGenerator(valid_config)
        ranked_df = generator.rank_anomalies(sample_scored_df)

        top_df = generator.apply_tags(ranked_df, top_n=20)

        assert "meets_min_spend" in top_df.columns
        assert "meets_min_transactions" in top_df.columns
        assert "rule_flagged" in top_df.columns

    def test_apply_tags_with_min_spend_rule(self, valid_config):
        """Test apply_tags() correctly applies minspend rule."""
        generator = ReportGenerator(valid_config)

        # Create DataFrame with known spend values
        test_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "anomaly_score": [-0.8, -0.5, -0.3],
                "total_spend": [50.0, 150.0, 200.0],  # minspend=100 in config
                "total_transactions": [10, 10, 10],
            }
        )

        top_df = generator.apply_tags(test_df, top_n=3)

        # C1 has spend < 100, should fail rule
        assert top_df.iloc[0]["meets_min_spend"] == False
        assert top_df.iloc[1]["meets_min_spend"] == True
        assert top_df.iloc[2]["meets_min_spend"] == True

    def test_apply_tags_with_min_transactions_rule(self, valid_config):
        """Test apply_tags() correctly applies mintransactions rule."""
        generator = ReportGenerator(valid_config)

        # Create DataFrame with known transaction counts
        test_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "anomaly_score": [-0.8, -0.5, -0.3],
                "total_spend": [200.0, 200.0, 200.0],
                "total_transactions": [3, 5, 10],  # mintransactions=5 in config
            }
        )

        top_df = generator.apply_tags(test_df, top_n=3)

        # C1 has transactions < 5, should fail rule
        assert top_df.iloc[0]["meets_min_transactions"] == False
        assert top_df.iloc[1]["meets_min_transactions"] == True
        assert top_df.iloc[2]["meets_min_transactions"] == True

    def test_apply_tags_rule_flagged_logic(self, valid_config):
        """Test apply_tags() sets rule_flagged correctly."""
        generator = ReportGenerator(valid_config)

        test_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3", "C4"],
                "anomaly_score": [-0.9, -0.7, -0.5, -0.3],
                "total_spend": [50.0, 150.0, 50.0, 200.0],  # minspend=100
                "total_transactions": [3, 10, 10, 10],  # mintransactions=5
            }
        )

        top_df = generator.apply_tags(test_df, top_n=4)

        # C1: fails both rules -> flagged
        assert top_df.iloc[0]["rule_flagged"] == True
        # C2: passes both rules -> not flagged
        assert top_df.iloc[1]["rule_flagged"] == False
        # C3: fails spend rule -> flagged
        assert top_df.iloc[2]["rule_flagged"] == True
        # C4: passes both rules -> not flagged
        assert top_df.iloc[3]["rule_flagged"] == False

    def test_apply_tags_without_spend_column(self, valid_config):
        """Test apply_tags() works when total_spend column missing."""
        generator = ReportGenerator(valid_config)

        test_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
                "total_transactions": [10, 10],
            }
        )

        top_df = generator.apply_tags(test_df, top_n=2)

        # Should default to True when column missing
        assert top_df["meets_min_spend"].all()
        assert not top_df["rule_flagged"].any()

    def test_apply_tags_without_transactions_column(self, valid_config):
        """Test apply_tags() works when total_transactions column missing."""
        generator = ReportGenerator(valid_config)

        test_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
                "total_spend": [200.0, 200.0],
            }
        )

        top_df = generator.apply_tags(test_df, top_n=2)

        # Should default to True when column missing
        assert top_df["meets_min_transactions"].all()
        assert not top_df["rule_flagged"].any()

    def test_apply_tags_with_invalid_top_n(self, valid_config, sample_scored_df):
        """Test apply_tags() raises error with invalid top_n."""
        generator = ReportGenerator(valid_config)
        ranked_df = generator.rank_anomalies(sample_scored_df)

        with pytest.raises(ReportGenerationError, match="top_n must be positive"):
            generator.apply_tags(ranked_df, top_n=0)

        with pytest.raises(ReportGenerationError, match="top_n must be positive"):
            generator.apply_tags(ranked_df, top_n=-5)

    def test_apply_tags_top_n_exceeds_length(
        self, valid_config, sample_scored_df, caplog
    ):
        """Test apply_tags() handles top_n exceeding DataFrame length."""
        generator = ReportGenerator(valid_config)
        ranked_df = generator.rank_anomalies(sample_scored_df)

        with caplog.at_level(logging.WARNING):
            top_df = generator.apply_tags(ranked_df, top_n=1000)

        # Should return all rows
        assert len(top_df) == len(ranked_df)
        assert "exceeds DataFrame length" in caplog.text

    def test_apply_tags_logs_flagged_count(
        self, valid_config, sample_scored_df, caplog
    ):
        """Test apply_tags() logs the count of flagged customers."""
        generator = ReportGenerator(valid_config)
        ranked_df = generator.rank_anomalies(sample_scored_df)

        with caplog.at_level(logging.INFO):
            top_df = generator.apply_tags(ranked_df, top_n=20)

        assert "Selected top 20 anomalies" in caplog.text
        assert "flagged by rules" in caplog.text

    def test_apply_tags_preserves_order(self, valid_config):
        """Test apply_tags() preserves ranking order from input."""
        generator = ReportGenerator(valid_config)

        test_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3", "C4", "C5"],
                "anomaly_score": [-0.9, -0.7, -0.5, -0.3, -0.1],
                "total_spend": [200.0, 200.0, 200.0, 200.0, 200.0],
                "total_transactions": [10, 10, 10, 10, 10],
            }
        )

        top_df = generator.apply_tags(test_df, top_n=3)

        # Should preserve order and select first 3
        assert top_df["customer_id"].tolist() == ["C1", "C2", "C3"]
        assert top_df["anomaly_score"].tolist() == [-0.9, -0.7, -0.5]

    def test_apply_tags_without_rules_config(self):
        """Test apply_tags() works when rules config is missing."""
        config = {"reporting": {"topnanomalies": 20}}  # No rules section
        generator = ReportGenerator(config)

        test_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
                "total_spend": [50.0, 200.0],
                "total_transactions": [3, 10],
            }
        )

        top_df = generator.apply_tags(test_df, top_n=2)

        # Should default all to True when no rules
        assert top_df["meets_min_spend"].all()
        assert top_df["meets_min_transactions"].all()
        assert not top_df["rule_flagged"].any()


class TestReportGeneratorJoinMccBreakdown:
    """Test join_mcc_breakdown method."""

    def test_join_mcc_breakdown_adds_mcc_columns(
        self, valid_config, sample_scored_df, sample_raw_df
    ):
        """Test join_mcc_breakdown() adds MCC spend and transaction columns."""
        generator = ReportGenerator(valid_config)
        ranked_df = generator.rank_anomalies(sample_scored_df)
        top_df = generator.apply_tags(ranked_df, top_n=10)

        result_df = generator.join_mcc_breakdown(top_df, sample_raw_df)

        # Should have MCC columns added
        mcc_cols = [col for col in result_df.columns if col.startswith("mcc_")]
        assert len(mcc_cols) > 0

        # Each MCC should have both spend and transaction columns
        mcc_codes = set()
        for col in mcc_cols:
            if col.endswith("_spend"):
                mcc_code = col.replace("mcc_", "").replace("_spend", "")
                mcc_codes.add(mcc_code)
                # Check corresponding transaction column exists
                assert f"mcc_{mcc_code}_transactions" in result_df.columns

    def test_join_mcc_breakdown_preserves_all_customers(
        self, valid_config, sample_scored_df, sample_raw_df
    ):
        """Test join_mcc_breakdown() preserves all customers from top_df."""
        generator = ReportGenerator(valid_config)
        ranked_df = generator.rank_anomalies(sample_scored_df)
        top_df = generator.apply_tags(ranked_df, top_n=10)

        result_df = generator.join_mcc_breakdown(top_df, sample_raw_df)

        # Should have same number of customers
        assert len(result_df) == len(top_df)
        assert set(result_df["customer_id"]) == set(top_df["customer_id"])

    def test_join_mcc_breakdown_aggregates_correctly(self, valid_config):
        """Test join_mcc_breakdown() correctly aggregates MCC data."""
        generator = ReportGenerator(valid_config)

        top_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
            }
        )

        raw_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C2"],
                "mcc": ["5411", "5411", "5812"],
                "spend_amount": [100.0, 50.0, 200.0],
                "transaction_count": [5, 3, 10],
            }
        )

        result_df = generator.join_mcc_breakdown(top_df, raw_df)

        # C1 should have aggregated spend for MCC 5411
        assert result_df.loc[result_df["customer_id"] == "C1", "mcc_5411_spend"].iloc[
            0
        ] == 150.0
        assert result_df.loc[
            result_df["customer_id"] == "C1", "mcc_5411_transactions"
        ].iloc[0] == 8

        # C2 should have spend for MCC 5812
        assert result_df.loc[result_df["customer_id"] == "C2", "mcc_5812_spend"].iloc[
            0
        ] == 200.0
        assert result_df.loc[
            result_df["customer_id"] == "C2", "mcc_5812_transactions"
        ].iloc[0] == 10

    def test_join_mcc_breakdown_fills_missing_mccs_with_zero(self, valid_config):
        """Test join_mcc_breakdown() fills missing MCC values with 0."""
        generator = ReportGenerator(valid_config)

        top_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
            }
        )

        raw_df = pd.DataFrame(
            {
                "customer_id": ["C1"],  # Only C1 has transactions
                "mcc": ["5411"],
                "spend_amount": [100.0],
                "transaction_count": [5],
            }
        )

        result_df = generator.join_mcc_breakdown(top_df, raw_df)

        # C1 should have MCC 5411 data
        assert result_df.loc[result_df["customer_id"] == "C1", "mcc_5411_spend"].iloc[
            0
        ] == 100.0

        # C2 should have 0 for MCC 5411 (no transactions)
        assert result_df.loc[result_df["customer_id"] == "C2", "mcc_5411_spend"].iloc[
            0
        ] == 0.0

    def test_join_mcc_breakdown_with_no_matching_customers(
        self, valid_config, caplog
    ):
        """Test join_mcc_breakdown() handles no matching customers."""
        generator = ReportGenerator(valid_config)

        top_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
            }
        )

        # Raw data has different customers
        raw_df = pd.DataFrame(
            {
                "customer_id": ["C3", "C4"],
                "mcc": ["5411", "5812"],
                "spend_amount": [100.0, 200.0],
                "transaction_count": [5, 10],
            }
        )

        with caplog.at_level(logging.WARNING):
            result_df = generator.join_mcc_breakdown(top_df, raw_df)

        # Should return top_df unchanged
        assert len(result_df) == len(top_df)
        assert list(result_df.columns) == list(top_df.columns)
        assert "No matching transaction data found" in caplog.text

    def test_join_mcc_breakdown_missing_required_columns(self, valid_config):
        """Test join_mcc_breakdown() raises error with missing columns."""
        generator = ReportGenerator(valid_config)

        top_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
            }
        )

        # Missing 'mcc' column
        invalid_raw_df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "spend_amount": [100.0],
                "transaction_count": [5],
            }
        )

        with pytest.raises(
            ReportGenerationError, match="raw_df missing required columns.*mcc"
        ):
            generator.join_mcc_breakdown(top_df, invalid_raw_df)

    def test_join_mcc_breakdown_handles_multiple_mccs_per_customer(
        self, valid_config
    ):
        """Test join_mcc_breakdown() handles customers with multiple MCCs."""
        generator = ReportGenerator(valid_config)

        top_df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "anomaly_score": [-0.8],
            }
        )

        raw_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C1"],
                "mcc": ["5411", "5812", "5999"],
                "spend_amount": [100.0, 200.0, 50.0],
                "transaction_count": [5, 10, 2],
            }
        )

        result_df = generator.join_mcc_breakdown(top_df, raw_df)

        # Should have columns for all 3 MCCs
        assert "mcc_5411_spend" in result_df.columns
        assert "mcc_5812_spend" in result_df.columns
        assert "mcc_5999_spend" in result_df.columns

        # Values should match
        assert result_df["mcc_5411_spend"].iloc[0] == 100.0
        assert result_df["mcc_5812_spend"].iloc[0] == 200.0
        assert result_df["mcc_5999_spend"].iloc[0] == 50.0

    def test_join_mcc_breakdown_logs_execution(
        self, valid_config, sample_scored_df, sample_raw_df, caplog
    ):
        """Test join_mcc_breakdown() logs execution details."""
        generator = ReportGenerator(valid_config)
        ranked_df = generator.rank_anomalies(sample_scored_df)
        top_df = generator.apply_tags(ranked_df, top_n=10)

        with caplog.at_level(logging.INFO):
            result_df = generator.join_mcc_breakdown(top_df, sample_raw_df)

        assert "Joined MCC breakdown for" in caplog.text
        assert "added" in caplog.text
        assert "MCC columns" in caplog.text

    def test_join_mcc_breakdown_preserves_original_columns(
        self, valid_config, sample_scored_df, sample_raw_df
    ):
        """Test join_mcc_breakdown() preserves all original columns from top_df."""
        generator = ReportGenerator(valid_config)
        ranked_df = generator.rank_anomalies(sample_scored_df)
        top_df = generator.apply_tags(ranked_df, top_n=10)

        original_cols = set(top_df.columns)

        result_df = generator.join_mcc_breakdown(top_df, sample_raw_df)

        # All original columns should still exist
        assert original_cols.issubset(set(result_df.columns))

    def test_join_mcc_breakdown_filters_to_top_customers_only(self, valid_config):
        """Test join_mcc_breakdown() only processes top N customers."""
        generator = ReportGenerator(valid_config)

        top_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
            }
        )

        # Raw data includes additional customers not in top_df
        raw_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3", "C4"],
                "mcc": ["5411", "5812", "5411", "5999"],
                "spend_amount": [100.0, 200.0, 300.0, 400.0],
                "transaction_count": [5, 10, 15, 20],
            }
        )

        result_df = generator.join_mcc_breakdown(top_df, raw_df)

        # Should only have 2 customers from top_df
        assert len(result_df) == 2
        assert set(result_df["customer_id"]) == {"C1", "C2"}


class TestReportGeneratorExportCsv:
    """Test export_csv method."""

    def test_export_csv_creates_file(self, valid_config, tmp_path):
        """Test export_csv() creates CSV file at specified path."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "anomaly_score": [-0.8, -0.5, -0.3],
                "total_spend": [100.0, 200.0, 300.0],
            }
        )

        output_path = tmp_path / "test_report.csv"
        result_path = generator.export_csv(report_df, output_path)

        assert Path(result_path).exists()
        assert Path(result_path) == output_path

    def test_export_csv_creates_parent_directory(self, valid_config, tmp_path):
        """Test export_csv() creates parent directories if they don't exist."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
            }
        )

        output_path = tmp_path / "nested" / "dir" / "report.csv"
        result_path = generator.export_csv(report_df, output_path)

        assert Path(result_path).exists()
        assert Path(result_path).parent.exists()

    def test_export_csv_content_format(self, valid_config, tmp_path):
        """Test export_csv() produces correctly formatted CSV content."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8123, -0.5678],
                "total_spend": [123.456789, 987.654321],
            }
        )

        output_path = tmp_path / "report.csv"
        generator.export_csv(report_df, output_path)

        # Read back and verify content
        result_df = pd.read_csv(output_path)

        assert len(result_df) == 2
        assert list(result_df.columns) == ["customer_id", "anomaly_score", "total_spend"]
        assert result_df["customer_id"].tolist() == ["C1", "C2"]

        # Check float formatting (4 decimal places)
        with open(output_path) as f:
            content = f.read()
            assert "-0.8123" in content
            assert "-0.5678" in content

    def test_export_csv_no_index(self, valid_config, tmp_path):
        """Test export_csv() exports without index column."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
            }
        )

        output_path = tmp_path / "report.csv"
        generator.export_csv(report_df, output_path)

        # Read first line to check headers
        with open(output_path) as f:
            first_line = f.readline().strip()
            # Should not have unnamed index column
            assert first_line == "customer_id,anomaly_score"

    def test_export_csv_utf8_encoding(self, valid_config, tmp_path):
        """Test export_csv() uses UTF-8 encoding."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
                "notes": ["café", "naïve"],
            }
        )

        output_path = tmp_path / "report.csv"
        generator.export_csv(report_df, output_path)

        # Read with UTF-8 encoding
        result_df = pd.read_csv(output_path, encoding="utf-8")
        assert result_df["notes"].tolist() == ["café", "naïve"]

    def test_export_csv_handles_many_columns(self, valid_config, tmp_path):
        """Test export_csv() handles DataFrames with many columns."""
        generator = ReportGenerator(valid_config)

        # Create DataFrame with many MCC columns
        data = {"customer_id": ["C1", "C2"], "anomaly_score": [-0.8, -0.5]}
        
        # Add 50 MCC columns
        for i in range(50):
            data[f"mcc_{5000+i}_spend"] = [100.0 * i, 200.0 * i]

        report_df = pd.DataFrame(data)

        output_path = tmp_path / "report.csv"
        result_path = generator.export_csv(report_df, output_path)

        # Verify file was created and has correct columns
        result_df = pd.read_csv(result_path)
        assert len(result_df.columns) == 52  # 2 base + 50 MCC

    def test_export_csv_returns_path_string(self, valid_config, tmp_path):
        """Test export_csv() returns path as string."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "anomaly_score": [-0.8],
            }
        )

        output_path = tmp_path / "report.csv"
        result_path = generator.export_csv(report_df, output_path)

        assert isinstance(result_path, str)
        assert result_path == str(output_path)

    def test_export_csv_accepts_string_path(self, valid_config, tmp_path):
        """Test export_csv() accepts path as string."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "anomaly_score": [-0.8],
            }
        )

        output_path = str(tmp_path / "report.csv")
        result_path = generator.export_csv(report_df, output_path)

        assert Path(result_path).exists()

    def test_export_csv_logs_execution(self, valid_config, tmp_path, caplog):
        """Test export_csv() logs export details."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "anomaly_score": [-0.8, -0.5, -0.3],
            }
        )

        output_path = tmp_path / "report.csv"

        with caplog.at_level(logging.INFO):
            generator.export_csv(report_df, output_path)

        assert "Exported CSV report" in caplog.text
        assert "3 rows" in caplog.text
        assert "bytes" in caplog.text

    def test_export_csv_overwrites_existing_file(self, valid_config, tmp_path):
        """Test export_csv() overwrites existing file."""
        generator = ReportGenerator(valid_config)

        output_path = tmp_path / "report.csv"

        # Create initial file
        report_df1 = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "anomaly_score": [-0.8],
            }
        )
        generator.export_csv(report_df1, output_path)

        # Overwrite with new data
        report_df2 = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "anomaly_score": [-0.9, -0.7, -0.5],
            }
        )
        generator.export_csv(report_df2, output_path)

        # Verify new data
        result_df = pd.read_csv(output_path)
        assert len(result_df) == 3

    def test_export_csv_with_empty_dataframe(self, valid_config, tmp_path):
        """Test export_csv() handles empty DataFrame."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": [],
                "anomaly_score": [],
            }
        )

        output_path = tmp_path / "report.csv"
        result_path = generator.export_csv(report_df, output_path)

        # Should create file with headers only
        assert Path(result_path).exists()
        result_df = pd.read_csv(result_path)
        assert len(result_df) == 0
        assert list(result_df.columns) == ["customer_id", "anomaly_score"]


class TestReportGeneratorExportSummaryJson:
    """Test export_summary_json method."""

    def test_export_summary_json_creates_file(self, valid_config, tmp_path):
        """Test export_summary_json() creates JSON file at specified path."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "anomaly_score": [-0.8, -0.5, -0.3],
                "anomaly_label": [-1, -1, 1],
            }
        )

        output_path = tmp_path / "summary.json"
        result_path = generator.export_summary_json(
            report_df, "2025-11-18", output_path
        )

        assert Path(result_path).exists()
        assert Path(result_path) == output_path

    def test_export_summary_json_structure(self, valid_config, tmp_path):
        """Test export_summary_json() produces correctly structured JSON."""
        import json

        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "anomaly_score": [-0.8, -0.5, -0.3],
                "anomaly_label": [-1, -1, 1],
            }
        )

        output_path = tmp_path / "summary.json"
        generator.export_summary_json(report_df, "2025-11-18", output_path)

        with open(output_path) as f:
            summary = json.load(f)

        # Check top-level keys
        assert "metadata" in summary
        assert "statistics" in summary
        assert "top_anomalies" in summary

        # Check metadata structure
        assert "reporting_week" in summary["metadata"]
        assert "generated_at" in summary["metadata"]
        assert "top_n_anomalies" in summary["metadata"]

        # Check statistics structure
        assert "total_customers" in summary["statistics"]
        assert "anomaly_count" in summary["statistics"]
        assert "rule_flagged_count" in summary["statistics"]
        assert "anomaly_score" in summary["statistics"]

        # Check top_anomalies structure
        assert "customer_ids" in summary["top_anomalies"]

    def test_export_summary_json_metadata_values(self, valid_config, tmp_path):
        """Test export_summary_json() includes correct metadata."""
        import json

        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
                "anomaly_label": [-1, -1],
            }
        )

        output_path = tmp_path / "summary.json"
        generator.export_summary_json(report_df, "2025-11-18", output_path)

        with open(output_path) as f:
            summary = json.load(f)

        assert summary["metadata"]["reporting_week"] == "2025-11-18"
        assert summary["metadata"]["top_n_anomalies"] == 20  # From config
        assert "T" in summary["metadata"]["generated_at"]  # ISO format

    def test_export_summary_json_statistics_values(self, valid_config, tmp_path):
        """Test export_summary_json() calculates correct statistics."""
        import json

        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3", "C4"],
                "anomaly_score": [-0.8, -0.5, -0.3, 0.2],
                "anomaly_label": [-1, -1, -1, 1],
                "rule_flagged": [True, False, True, False],
            }
        )

        output_path = tmp_path / "summary.json"
        generator.export_summary_json(report_df, "2025-11-18", output_path)

        with open(output_path) as f:
            summary = json.load(f)

        stats = summary["statistics"]
        assert stats["total_customers"] == 4
        assert stats["anomaly_count"] == 3  # 3 with label -1
        assert stats["rule_flagged_count"] == 2  # 2 with rule_flagged=True

        # Score statistics
        assert stats["anomaly_score"]["min"] == -0.8
        assert stats["anomaly_score"]["max"] == 0.2
        assert stats["anomaly_score"]["mean"] == pytest.approx(-0.35, abs=0.01)
        assert stats["anomaly_score"]["median"] == pytest.approx(-0.4, abs=0.01)

    def test_export_summary_json_customer_ids(self, valid_config, tmp_path):
        """Test export_summary_json() includes correct customer IDs."""
        import json

        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003"],
                "anomaly_score": [-0.9, -0.7, -0.5],
                "anomaly_label": [-1, -1, -1],
            }
        )

        output_path = tmp_path / "summary.json"
        generator.export_summary_json(report_df, "2025-11-18", output_path)

        with open(output_path) as f:
            summary = json.load(f)

        assert summary["top_anomalies"]["customer_ids"] == ["C001", "C002", "C003"]

    def test_export_summary_json_without_anomaly_label(self, valid_config, tmp_path):
        """Test export_summary_json() handles missing anomaly_label column."""
        import json

        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
            }
        )

        output_path = tmp_path / "summary.json"
        generator.export_summary_json(report_df, "2025-11-18", output_path)

        with open(output_path) as f:
            summary = json.load(f)

        # Should default to 0 when column missing
        assert summary["statistics"]["anomaly_count"] == 0

    def test_export_summary_json_without_rule_flagged(self, valid_config, tmp_path):
        """Test export_summary_json() handles missing rule_flagged column."""
        import json

        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
                "anomaly_label": [-1, -1],
            }
        )

        output_path = tmp_path / "summary.json"
        generator.export_summary_json(report_df, "2025-11-18", output_path)

        with open(output_path) as f:
            summary = json.load(f)

        # Should default to 0 when column missing
        assert summary["statistics"]["rule_flagged_count"] == 0

    def test_export_summary_json_creates_parent_directory(
        self, valid_config, tmp_path
    ):
        """Test export_summary_json() creates parent directories."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "anomaly_score": [-0.8],
                "anomaly_label": [-1],
            }
        )

        output_path = tmp_path / "nested" / "dir" / "summary.json"
        generator.export_summary_json(report_df, "2025-11-18", output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_export_summary_json_pretty_formatting(self, valid_config, tmp_path):
        """Test export_summary_json() uses pretty formatting (indented)."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "anomaly_score": [-0.8],
                "anomaly_label": [-1],
            }
        )

        output_path = tmp_path / "summary.json"
        generator.export_summary_json(report_df, "2025-11-18", output_path)

        with open(output_path) as f:
            content = f.read()

        # Pretty formatted JSON should have newlines and indentation
        assert "\n" in content
        assert "  " in content  # 2-space indent

    def test_export_summary_json_utf8_encoding(self, valid_config, tmp_path):
        """Test export_summary_json() uses UTF-8 encoding."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["Café", "Naïve"],
                "anomaly_score": [-0.8, -0.5],
                "anomaly_label": [-1, -1],
            }
        )

        output_path = tmp_path / "summary.json"
        generator.export_summary_json(report_df, "2025-11-18", output_path)

        with open(output_path, encoding="utf-8") as f:
            import json
            summary = json.load(f)

        assert summary["top_anomalies"]["customer_ids"] == ["Café", "Naïve"]

    def test_export_summary_json_logs_execution(
        self, valid_config, tmp_path, caplog
    ):
        """Test export_summary_json() logs execution details."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [-0.8, -0.5],
                "anomaly_label": [-1, -1],
            }
        )

        output_path = tmp_path / "summary.json"

        with caplog.at_level(logging.INFO):
            generator.export_summary_json(report_df, "2025-11-18", output_path)

        assert "Exported JSON summary" in caplog.text
        assert "bytes" in caplog.text

    def test_export_summary_json_returns_path_string(self, valid_config, tmp_path):
        """Test export_summary_json() returns path as string."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "anomaly_score": [-0.8],
                "anomaly_label": [-1],
            }
        )

        output_path = tmp_path / "summary.json"
        result_path = generator.export_summary_json(
            report_df, "2025-11-18", output_path
        )

        assert isinstance(result_path, str)
        assert result_path == str(output_path)

    def test_export_summary_json_accepts_string_path(self, valid_config, tmp_path):
        """Test export_summary_json() accepts path as string."""
        generator = ReportGenerator(valid_config)

        report_df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "anomaly_score": [-0.8],
                "anomaly_label": [-1],
            }
        )

        output_path = str(tmp_path / "summary.json")
        result_path = generator.export_summary_json(
            report_df, "2025-11-18", output_path
        )

        assert Path(result_path).exists()

    def test_export_summary_json_missing_customer_id_raises_error(
        self, valid_config, tmp_path
    ):
        """Test export_summary_json() raises error when customer_id column is missing."""
        generator = ReportGenerator(valid_config)

        # DataFrame without customer_id column
        report_df = pd.DataFrame(
            {
                "anomaly_score": [-0.8, -0.5],
                "anomaly_label": [-1, -1],
            }
        )

        output_path = tmp_path / "summary.json"

        with pytest.raises(
            ReportGenerationError,
            match="Required column 'customer_id' not found in report DataFrame",
        ):
            generator.export_summary_json(report_df, "2025-11-18", output_path)

    def test_export_summary_json_missing_anomaly_score_raises_error(
        self, valid_config, tmp_path
    ):
        """Test export_summary_json() raises error when anomaly_score column is missing."""
        generator = ReportGenerator(valid_config)

        # DataFrame without anomaly_score column
        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_label": [-1, -1],
            }
        )

        output_path = tmp_path / "summary.json"

        with pytest.raises(
            ReportGenerationError,
            match="Required column 'anomaly_score' not found in report DataFrame",
        ):
            generator.export_summary_json(report_df, "2025-11-18", output_path)

    def test_export_summary_json_all_nan_anomaly_score_raises_error(
        self, valid_config, tmp_path
    ):
        """Test export_summary_json() raises error when anomaly_score has only NaN values."""
        generator = ReportGenerator(valid_config)

        # DataFrame with all NaN anomaly_score values
        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "anomaly_score": [float("nan"), float("nan")],
                "anomaly_label": [-1, -1],
            }
        )

        output_path = tmp_path / "summary.json"

        with pytest.raises(
            ReportGenerationError,
            match="Column 'anomaly_score' contains no valid \\(non-NaN\\) values",
        ):
            generator.export_summary_json(report_df, "2025-11-18", output_path)

    def test_export_summary_json_handles_partial_nan_anomaly_score(
        self, valid_config, tmp_path
    ):
        """Test export_summary_json() handles mix of valid and NaN anomaly_score values."""
        import json

        generator = ReportGenerator(valid_config)

        # DataFrame with some NaN anomaly_score values
        report_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3", "C4"],
                "anomaly_score": [-0.8, float("nan"), -0.3, float("nan")],
                "anomaly_label": [-1, -1, -1, 1],
            }
        )

        output_path = tmp_path / "summary.json"
        generator.export_summary_json(report_df, "2025-11-18", output_path)

        with open(output_path) as f:
            summary = json.load(f)

        # Statistics should only include non-NaN values (-0.8, -0.3)
        stats = summary["statistics"]["anomaly_score"]
        assert stats["min"] == -0.8
        assert stats["max"] == -0.3
        assert stats["mean"] == pytest.approx(-0.55, abs=0.01)
        assert stats["median"] == pytest.approx(-0.55, abs=0.01)

        # Should not include MCC data from C3/C4
        # (but might have their MCC codes if C1/C2 also have transactions there)
