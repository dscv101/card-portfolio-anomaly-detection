"""Unit tests for ReportGenerator class.

Tests cover:
- Initialization and configuration validation
- Input DataFrame validation
- Report generation orchestration
- Error handling for missing configuration
- Error handling for invalid inputs
"""

import logging

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
            generator.apply_tags(ranked_df, top_n=20)

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
