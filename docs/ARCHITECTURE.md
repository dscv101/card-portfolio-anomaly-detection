# Architecture Documentation: Card Portfolio Anomaly Detection System

**Version:** 1.0.0  
**Last Updated:** 2025-11-21  
**Status:** Production  
**Owner:** Data Science Team

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Principles](#architecture-principles)
3. [Component Design](#component-design)
4. [Data Flow](#data-flow)
5. [Configuration System](#configuration-system)
6. [Error Handling](#error-handling)
7. [Logging & Observability](#logging--observability)
8. [Testing Strategy](#testing-strategy)
9. [Deployment Architecture](#deployment-architecture)
10. [Security Considerations](#security-considerations)

---

## System Overview

### Purpose

The Card Portfolio Anomaly Detection System is a batch-processing pipeline that identifies unusual customer behavior patterns in credit card transaction data using unsupervised machine learning.

### Key Design Goals

1. **Spec-Driven**: All behavior controlled by configuration files, not code constants
2. **Observable**: Comprehensive logging and monitoring at every stage
3. **Testable**: High test coverage with unit, integration, and E2E tests
4. **Maintainable**: Clear separation of concerns, modular architecture
5. **Deterministic**: Same inputs + same config = same outputs (reproducibility)

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.9+ | Core implementation |
| ML Framework | scikit-learn | 1.3+ | Isolation Forest algorithm |
| Data Processing | pandas | 2.0+ | Data manipulation |
| Configuration | PyYAML | 6.0+ | Config file parsing |
| Testing | pytest | 7.4+ | Test framework |
| Linting | Ruff, Flake8 | Latest | Code quality |
| Formatting | Black | Latest | Code style |
| Type Checking | mypy | 1.0+ | Static type validation |

---

## Architecture Principles

### 1. Separation of Concerns

The system is divided into distinct layers with clear responsibilities:

```
┌─────────────────────────────────────┐
│         Orchestration Layer         │
│     (main.py, cli.py)              │
└─────────────────────────────────────┘
           ▼          ▼          ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐
│  Data Layer  │ │ Features │ │  Models  │
│              │ │  Layer   │ │  Layer   │
└──────────────┘ └──────────┘ └──────────┘
           ▼          ▼          ▼
┌─────────────────────────────────────┐
│         Reporting Layer             │
│     (generator.py)                  │
└─────────────────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│         Utilities Layer             │
│  (logging, config, exceptions)      │
└─────────────────────────────────────┘
```

### 2. Configuration Over Code

All behavioral parameters are externalized:
- **Model Parameters**: `config/modelconfig.yaml`
- **Data Sources**: `config/dataconfig.yaml`
- **Feature Engineering**: Defined in config, not hardcoded
- **Thresholds**: All limits configurable

### 3. Fail-Fast Philosophy

- Validate inputs at system boundaries
- Use type hints and runtime type checking
- Raise explicit exceptions with actionable messages
- No silent failures or data corruption

### 4. Immutability Where Possible

- Data transformations return new objects
- Configuration objects are read-only after loading
- State mutations are explicit and logged

---

## Component Design

### Data Layer (`src/data/`)

**Purpose**: Load and validate raw transaction data

**Modules**:

#### `loader.py`
```python
class DataLoader:
    """Loads customer transaction data from various sources."""
    
    def load(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from configured source (CSV, Parquet, DB, etc.)"""
        
    def filter(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply business logic filters (min transaction count, date ranges)"""
```

**Responsibilities**:
- Read from multiple source types (CSV, Parquet, S3, databases)
- Apply data filters (date ranges, customer segments)
- Handle missing files gracefully
- Log data loading statistics

#### `validator.py`
```python
class DataValidator:
    """Validates data schema and business rules."""
    
    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict) -> None:
        """Ensure required columns exist with correct types"""
        
    def validate_business_rules(self, df: pd.DataFrame, rules: Dict) -> None:
        """Check data quality constraints (no negatives, valid ranges)"""
```

**Validation Checks**:
- Required columns present
- Correct data types
- No negative values in amount fields
- Valid date formats
- Reasonable value ranges (configurable thresholds)

---

### Features Layer (`src/features/`)

**Purpose**: Transform raw data into ML-ready features

#### `builder.py`
```python
class FeatureBuilder:
    """Builds behavioral features from transaction data."""
    
    def build_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transactions to customer level"""
        
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate diversity, concentration, velocity metrics"""
```

**Feature Categories**:

1. **Volume Features**:
   - `total_spend`: Sum of all transactions
   - `transaction_count`: Number of transactions
   - `avg_ticket_size`: Mean transaction amount

2. **Diversity Features**:
   - `mcc_diversity`: Number of unique merchant categories
   - `mcc_concentration`: Spend share in top category
   - `mcc_entropy`: Shannon entropy of category distribution

3. **Velocity Features** (optional):
   - `spend_growth_rate`: Week-over-week change
   - `transaction_frequency`: Transactions per day

**Design Decisions**:
- All features calculated from single reporting period (cross-sectional)
- No time-series features in v1.0 (future enhancement)
- Feature selection configurable via `modelconfig.yaml`

---

### Models Layer (`src/models/`)

**Purpose**: Train and apply anomaly detection models

#### `scorer.py`
```python
class AnomalyScorer:
    """Trains Isolation Forest and scores customers."""
    
    def train(self, features: pd.DataFrame, config: Dict) -> None:
        """Fit Isolation Forest model to feature data"""
        
    def score(self, features: pd.DataFrame) -> pd.Series:
        """Generate anomaly scores (-1 to 1, lower = more anomalous)"""
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Extract which features contributed to anomaly detection"""
```

**Algorithm: Isolation Forest**

Why Isolation Forest?
- ✅ Unsupervised (no labeled training data required)
- ✅ Fast training and prediction
- ✅ Handles high-dimensional data well
- ✅ Robust to outliers in features
- ✅ Interpretable scores

**Key Parameters** (in `modelconfig.yaml`):
```yaml
model:
  algorithm: "IsolationForest"
  contamination: 0.05      # Expected % of anomalies
  n_estimators: 100        # Number of trees
  max_samples: 256         # Samples per tree
  random_state: 42         # For reproducibility
```

**Reproducibility**:
- Fixed `random_state` ensures deterministic results
- Feature order preserved via sorted column names
- Model serialization for exact replication

---

### Reporting Layer (`src/reporting/`)

**Purpose**: Generate human-readable outputs

#### `generator.py`
```python
class ReportGenerator:
    """Creates CSV and HTML reports from anomaly scores."""
    
    def generate_csv(self, anomalies: pd.DataFrame, output_path: str) -> None:
        """Export ranked anomalies to CSV"""
        
    def generate_html(self, anomalies: pd.DataFrame, output_path: str) -> None:
        """Create visual dashboard with charts and tables"""
        
    def add_business_tags(self, anomalies: pd.DataFrame) -> pd.DataFrame:
        """Label anomalies as opportunities or concerns"""
```

**Report Components**:

1. **CSV Output**:
   - All detected anomalies with scores
   - Feature values for context
   - Business tags (opportunity/concern)

2. **HTML Dashboard**:
   - Summary statistics
   - Top 10 anomalies table
   - Feature importance charts
   - Opportunity vs. concern breakdown

---

### Utilities Layer (`src/utils/`)

#### `config_loader.py`
```python
class ConfigLoader:
    """Loads and validates YAML configuration files."""
    
    def load(self, path: str) -> Dict[str, Any]:
        """Parse YAML and validate structure"""
        
    def merge(self, base_config: Dict, overrides: Dict) -> Dict:
        """Support config inheritance and overrides"""
```

#### `logger.py`
```python
class StructuredLogger:
    """Provides JSON and text logging with context."""
    
    def info(self, message: str, context: Dict = None) -> None:
    def error(self, message: str, exception: Exception = None) -> None:
    def metric(self, name: str, value: float, unit: str) -> None:
```

**Logging Levels**:
- `DEBUG`: Detailed execution traces (not in production)
- `INFO`: Normal operation milestones
- `WARNING`: Recoverable issues
- `ERROR`: Failures that require attention
- `CRITICAL`: System-level failures

#### `exceptions.py`
```python
class DataValidationError(Exception):
    """Raised when data fails validation checks"""
    
class ConfigurationError(Exception):
    """Raised when config files are invalid or missing"""
    
class ModelError(Exception):
    """Raised when model training or prediction fails"""
```

---

## Data Flow

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────┐
│ 1. Data Loading                                     │
│    - Read from source (CSV/DB/S3)                   │
│    - Apply filters (date, segment)                  │
│    - Log: rows loaded, time elapsed                 │
└─────────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ 2. Data Validation                                  │
│    - Check schema (columns, types)                  │
│    - Validate business rules (no negatives, etc.)   │
│    - Fail fast if invalid                           │
└─────────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ 3. Feature Engineering                              │
│    - Aggregate to customer level                    │
│    - Calculate diversity, concentration metrics     │
│    - Log: feature statistics, null counts           │
└─────────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ 4. Model Training                                   │
│    - Fit Isolation Forest to features               │
│    - Validate model convergence                     │
│    - Extract feature importance                     │
└─────────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ 5. Anomaly Scoring                                  │
│    - Predict anomaly scores for all customers       │
│    - Rank by score (most anomalous first)           │
│    - Select top N for reporting                     │
└─────────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ 6. Report Generation                                │
│    - Add business tags (opportunity/concern)        │
│    - Generate CSV and HTML outputs                  │
│    - Log: output file paths                         │
└─────────────────────────────────────────────────────┘
```

### Typical Execution Time

| Stage | Time (10K customers) | Bottleneck |
|-------|---------------------|------------|
| Data Loading | 2-5 seconds | I/O (disk/network) |
| Validation | <1 second | CPU (pandas operations) |
| Feature Engineering | 3-8 seconds | CPU (aggregations) |
| Model Training | 5-10 seconds | CPU (scikit-learn) |
| Scoring | 1-2 seconds | CPU (tree traversal) |
| Reporting | 2-5 seconds | I/O (file writes) |
| **Total** | **15-30 seconds** | - |

---

## Configuration System

### Configuration File Structure

#### `config/modelconfig.yaml`
```yaml
model:
  algorithm: "IsolationForest"
  contamination: 0.05
  n_estimators: 100
  max_samples: 256
  random_state: 42

features:
  use:
    - "total_spend"
    - "transaction_count"
    - "avg_ticket_size"
    - "mcc_diversity"
    - "mcc_concentration"
  
  normalize: true
  normalization_method: "standard"  # or "minmax"

reporting:
  top_n: 20
  output_formats:
    - "csv"
    - "html"
  
  business_rules:
    opportunity_thresholds:
      high_spend_percentile: 90
      high_diversity_min: 10
    concern_thresholds:
      concentration_risk_threshold: 0.8
```

#### `config/dataconfig.yaml`
```yaml
data:
  source:
    type: "csv"  # or "parquet", "postgres", "s3"
    path: "datasamples/transactions.csv"
    
  schema:
    required_columns:
      - customer_id
      - reporting_week
      - mcc
      - spend_amount
      - transaction_count
    
    types:
      customer_id: "string"
      reporting_week: "string"
      mcc: "string"
      spend_amount: "float"
      transaction_count: "int"
  
  filters:
    min_transaction_count: 5
    max_days_ago: 30
    reporting_week: "latest"  # or "2025-W47"
  
  validation:
    allow_negatives: false
    max_spend_per_transaction: 100000
```

### Configuration Precedence

1. **Command-line arguments** (highest priority)
2. **Environment variables** (`ANOMALY_DETECTION_*`)
3. **Config files** (YAML)
4. **Default values** (lowest priority)

---

## Error Handling

### Exception Hierarchy

```
Exception
├── DataValidationError
│   ├── SchemaValidationError
│   └── BusinessRuleViolationError
├── ConfigurationError
│   ├── MissingConfigError
│   └── InvalidConfigError
└── ModelError
    ├── TrainingFailureError
    └── PredictionError
```

### Error Response Strategy

| Error Type | Action | User Message | Logging |
|------------|--------|--------------|---------|
| Data validation | Fail fast | "Invalid data: [specific issue]" | ERROR + context |
| Config missing | Fail fast | "Config file not found: [path]" | ERROR |
| Model warning | Continue with defaults | "Using default params" | WARNING |
| Report write failure | Retry 3x, then fail | "Cannot write report: [reason]" | ERROR + stacktrace |

### Graceful Degradation

- If HTML report fails, still generate CSV
- If feature importance extraction fails, report without it
- Log all degradations for troubleshooting

---

## Logging & Observability

### Log Formats

**Text Format** (human-readable):
```
2025-11-21 10:30:15 | INFO | DataLoader | Loaded 10245 records in 2.3s
2025-11-21 10:30:18 | INFO | FeatureBuilder | Built 8 features for 10245 customers
2025-11-21 10:30:25 | INFO | AnomalyScorer | Trained model, detected 512 anomalies
```

**JSON Format** (machine-parseable):
```json
{
  "timestamp": "2025-11-21T10:30:15Z",
  "level": "INFO",
  "module": "DataLoader",
  "message": "Loaded records",
  "context": {
    "record_count": 10245,
    "duration_seconds": 2.3,
    "source": "datasamples/transactions.csv"
  }
}
```

### Key Metrics Logged

- **Data Loading**: Record count, duration, source path
- **Validation**: Failed checks, invalid rows
- **Feature Engineering**: Feature count, null values, distributions
- **Model Training**: Contamination rate, training time, convergence status
- **Scoring**: Anomaly count, score distribution, runtime
- **Reporting**: Output file paths, record counts

### Monitoring Recommendations

For production deployments, monitor:
- **Execution Time**: Alert if >15 minutes
- **Anomaly Rate**: Alert if <1% or >10% (likely data issue)
- **Data Volume**: Alert if record count drops >20% week-over-week
- **Failures**: Alert on any ERROR-level logs

---

## Testing Strategy

### Test Pyramid

```
          ┌────────────┐
          │ E2E Tests  │  3 tests (full pipeline scenarios)
          └────────────┘
        ┌──────────────────┐
        │ Integration Tests │  12 tests (module interactions)
        └──────────────────┘
      ┌──────────────────────────┐
      │     Unit Tests           │  50+ tests (individual functions)
      └──────────────────────────┘
```

### Test Coverage

| Module | Unit Tests | Integration Tests | E2E Tests |
|--------|------------|-------------------|-----------|
| `data.loader` | 8 | 2 | - |
| `data.validator` | 10 | 1 | - |
| `features.builder` | 12 | 2 | - |
| `models.scorer` | 15 | 3 | - |
| `reporting.generator` | 8 | 2 | - |
| `utils.*` | 10 | - | - |
| **Pipeline** | - | 2 | 3 |
| **Total** | **63** | **12** | **3** |

**Target Coverage**: >80% line coverage, 100% of critical paths

### Test Types

**1. Unit Tests** (`tests/unit/`)
- Test individual functions in isolation
- Mock external dependencies (file I/O, network)
- Fast execution (<1s total)

**2. Integration Tests** (`tests/integration/`)
- Test module interactions (e.g., loader → validator)
- Use realistic test data (sample CSV files)
- Moderate execution time (<10s total)

**3. E2E Tests** (`tests/e2e/`)
- Test complete pipeline with real configurations
- Validate output files are generated correctly
- Slow execution (30-60s total)

**4. Specification Tests** (`tests/spectests/`)
- Verify behavior matches requirements doc
- Use requirement IDs (e.g., `test_REQ_1_1_1`)
- Traceability: Code ↔ Tests ↔ Requirements

---

## Deployment Architecture

### Local Execution

```bash
python main.py \
  --model-config config/modelconfig.yaml \
  --data-config config/dataconfig.yaml
```

### Scheduled Batch Job (Production)

**Option 1: Cron Job**
```bash
# Run every Monday at 9 AM
0 9 * * 1 cd /opt/anomaly-detection && ./venv/bin/python main.py --model-config config/modelconfig.yaml --data-config config/dataconfig.yaml
```

**Option 2: Airflow DAG**
```python
from airflow import DAG
from airflow.operators.bash import BashOperator

dag = DAG('anomaly_detection', schedule_interval='0 9 * * 1')

run_detection = BashOperator(
    task_id='run_anomaly_detection',
    bash_command='cd /opt/anomaly-detection && ./venv/bin/python main.py',
    dag=dag
)
```

**Option 3: Docker Container**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py", "--model-config", "config/modelconfig.yaml"]
```

### Infrastructure Requirements

| Environment | CPU | Memory | Disk | Estimated Cost |
|-------------|-----|--------|------|----------------|
| Development | 2 cores | 4 GB | 10 GB | Local machine |
| Staging | 2 cores | 8 GB | 20 GB | ~$50/month |
| Production | 4 cores | 16 GB | 50 GB | ~$150/month |

---

## Security Considerations

### Data Protection

- **PII Handling**: Customer IDs are hashed in reports (optional)
- **Access Control**: Config files define who can read source data
- **Encryption**: All data at rest encrypted (AWS S3, DB credentials in secrets manager)

### Secrets Management

```bash
# Store credentials in environment variables
export DB_PASSWORD=$(aws secretsmanager get-secret-value --secret-id prod/db/password)

# Or use .env file (not committed to Git)
# .env
DB_HOST=production.db.internal
DB_USER=anomaly_reader
DB_PASSWORD=<secret>
```

### Audit Trail

- Log all data accesses (who, when, what)
- Track configuration changes (Git history)
- Retain reports for compliance (90 days minimum)

---

## Performance Optimization

### Current Bottlenecks

1. **Feature aggregation** (40% of runtime)
   - Solution: Use `pandas.groupby` with optimized aggregations
   
2. **Model training** (30% of runtime)
   - Solution: Limit `n_estimators` or `max_samples`

3. **Data I/O** (20% of runtime)
   - Solution: Use Parquet instead of CSV, enable compression

### Scalability

| Data Volume | Execution Time | Recommended Instance |
|-------------|----------------|----------------------|
| <50K customers | <1 minute | 2 CPU, 4GB RAM |
| 50K-500K customers | 1-10 minutes | 4 CPU, 8GB RAM |
| 500K-5M customers | 10-60 minutes | 8 CPU, 16GB RAM |
| >5M customers | Distributed processing | Spark/Dask cluster |

### Future Enhancements

- **Parallel processing**: Use `joblib` or `multiprocessing` for feature engineering
- **Incremental updates**: Only reprocess new/changed customers
- **Distributed training**: Use Dask-ML for >1M customer portfolios

---

## Maintenance & Operations

### Routine Tasks

| Task | Frequency | Owner | Estimated Time |
|------|-----------|-------|----------------|
| Review logs for errors | Weekly | Data Engineer | 15 min |
| Validate output quality | Weekly | Data Analyst | 30 min |
| Update config based on feedback | Monthly | Data Science Team | 1 hour |
| Review model performance | Quarterly | Data Science Team | 4 hours |

### Upgrade Path

To upgrade the system:
1. Run full test suite: `pytest`
2. Update dependencies: `pip install -U -r requirements.txt`
3. Re-run integration tests
4. Deploy to staging first
5. Monitor for 1 week before production rollout

---

## References

- **Requirements**: [docs/REQUIREMENTS.md](REQUIREMENTS.md)
- **Task Breakdown**: [docs/tasks.md](tasks.md)
- **User Guide**: [docs/USER_GUIDE.md](USER_GUIDE.md)
- **API Reference**: [docs/API_REFERENCE.md](API_REFERENCE.md)
- **Coding Standards**: [AGENTS.md](../AGENTS.md)

---

**Document Maintained By**: Data Science Team  
**Last Review**: 2025-11-21  
**Next Review**: 2025-12-21

