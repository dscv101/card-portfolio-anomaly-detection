# API Reference: Card Portfolio Anomaly Detection System

**Version:** 1.0.0  
**Last Updated:** 2025-11-21  
**Target Audience:** Developers, Data Engineers

---

## Table of Contents

1. [Data Layer API](#data-layer-api)
2. [Features Layer API](#features-layer-api)
3. [Models Layer API](#models-layer-api)
4. [Reporting Layer API](#reporting-layer-api)
5. [Utilities API](#utilities-api)
6. [CLI Reference](#cli-reference)
7. [Configuration Schema](#configuration-schema)

---

## Data Layer API

### `src.data.loader.DataLoader`

Loads customer transaction data from various sources.

#### Constructor

```python
DataLoader(config: Dict[str, Any])
```

**Parameters:**
- `config` (dict): Configuration dictionary from `dataconfig.yaml`

**Example:**
```python
from src.data.loader import DataLoader

config = {
    "source": {
        "type": "csv",
        "path": "datasamples/transactions.csv"
    }
}
loader = DataLoader(config)
```

---

#### `load() -> pd.DataFrame`

Load data from the configured source.

**Returns:**
- `pd.DataFrame`: Raw transaction data

**Raises:**
- `FileNotFoundError`: If source file doesn't exist
- `DataValidationError`: If data cannot be parsed

**Example:**
```python
df = loader.load()
print(f"Loaded {len(df)} records")
```

**Expected DataFrame Schema:**
```python
{
    'customer_id': str,      # Unique customer identifier
    'reporting_week': str,   # ISO week format (e.g., "2025-W47")
    'mcc': str,              # Merchant Category Code
    'spend_amount': float,   # Transaction amount
    'transaction_count': int # Number of transactions
}
```

---

#### `filter(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame`

Apply business logic filters to data.

**Parameters:**
- `df` (pd.DataFrame): Input data
- `filters` (dict): Filter criteria from config

**Returns:**
- `pd.DataFrame`: Filtered data

**Example:**
```python
filters = {
    "min_transaction_count": 5,
    "reporting_week": "2025-W47"
}
filtered_df = loader.filter(df, filters)
```

---

### `src.data.validator.DataValidator`

Validates data schema and business rules.

#### Constructor

```python
DataValidator(config: Dict[str, Any])
```

**Parameters:**
- `config` (dict): Validation rules from `dataconfig.yaml`

---

#### `validate_schema(df: pd.DataFrame) -> None`

Validate that DataFrame matches expected schema.

**Parameters:**
- `df` (pd.DataFrame): Data to validate

**Raises:**
- `SchemaValidationError`: If required columns missing or types incorrect

**Example:**
```python
from src.data.validator import DataValidator

validator = DataValidator(config)
try:
    validator.validate_schema(df)
    print("Schema validation passed")
except SchemaValidationError as e:
    print(f"Validation failed: {e}")
```

---

#### `validate_business_rules(df: pd.DataFrame) -> None`

Validate business constraints (no negatives, reasonable ranges).

**Parameters:**
- `df` (pd.DataFrame): Data to validate

**Raises:**
- `BusinessRuleViolationError`: If rules violated

**Example:**
```python
validator.validate_business_rules(df)
```

**Business Rules Checked:**
- No negative values in `spend_amount` or `transaction_count`
- `spend_amount` within reasonable range (0 to `max_spend_per_transaction`)
- Valid date formats

---

## Features Layer API

### `src.features.builder.FeatureBuilder`

Transforms transaction data into ML-ready features.

#### Constructor

```python
FeatureBuilder(config: Dict[str, Any])
```

**Parameters:**
- `config` (dict): Feature engineering configuration from `modelconfig.yaml`

---

#### `build_customer_features(df: pd.DataFrame) -> pd.DataFrame`

Aggregate transactions to customer level.

**Parameters:**
- `df` (pd.DataFrame): Transaction-level data

**Returns:**
- `pd.DataFrame`: Customer-level aggregated features

**Example:**
```python
from src.features.builder import FeatureBuilder

builder = FeatureBuilder(config)
features = builder.build_customer_features(df)
```

**Output Schema:**
```python
{
    'customer_id': str,
    'total_spend': float,
    'transaction_count': int,
    'avg_ticket_size': float,
    'mcc_diversity': int,
    'mcc_concentration': float
}
```

---

#### `add_derived_features(df: pd.DataFrame) -> pd.DataFrame`

Calculate advanced metrics (diversity, concentration).

**Parameters:**
- `df` (pd.DataFrame): Customer features

**Returns:**
- `pd.DataFrame`: Features with derived metrics added

**Derived Features:**
- `mcc_diversity`: Count of unique merchant categories
- `mcc_concentration`: Spend share in top category (Herfindahl index)
- `mcc_entropy`: Shannon entropy of category distribution

**Example:**
```python
enhanced_features = builder.add_derived_features(features)
```

---

#### `normalize_features(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame`

Normalize feature values for model input.

**Parameters:**
- `df` (pd.DataFrame): Raw features
- `method` (str): Normalization method ("standard" or "minmax")

**Returns:**
- `pd.DataFrame`: Normalized features

**Example:**
```python
normalized = builder.normalize_features(features, method="standard")
```

**Normalization Methods:**
- `"standard"`: Z-score normalization (mean=0, std=1)
- `"minmax"`: Scale to [0, 1] range

---

## Models Layer API

### `src.models.scorer.AnomalyScorer`

Trains Isolation Forest and scores customer anomalies.

#### Constructor

```python
AnomalyScorer(config: Dict[str, Any])
```

**Parameters:**
- `config` (dict): Model configuration from `modelconfig.yaml`

---

#### `train(features: pd.DataFrame) -> None`

Fit Isolation Forest model to feature data.

**Parameters:**
- `features` (pd.DataFrame): Normalized customer features

**Raises:**
- `ModelError`: If training fails

**Example:**
```python
from src.models.scorer import AnomalyScorer

scorer = AnomalyScorer(config)
scorer.train(features)
```

**Model Parameters** (from config):
```yaml
model:
  algorithm: "IsolationForest"
  contamination: 0.05
  n_estimators: 100
  max_samples: 256
  random_state: 42
```

---

#### `score(features: pd.DataFrame) -> pd.Series`

Predict anomaly scores for customers.

**Parameters:**
- `features` (pd.DataFrame): Customer features to score

**Returns:**
- `pd.Series`: Anomaly scores (index=customer_id, values=score)

**Score Interpretation:**
- **< -0.2**: Strong anomaly
- **-0.1 to -0.2**: Moderate anomaly
- **> -0.1**: Normal behavior

**Example:**
```python
scores = scorer.score(features)
print(f"Most anomalous customer: {scores.idxmin()} (score: {scores.min():.3f})")
```

---

#### `get_feature_importance() -> pd.DataFrame`

Extract which features contributed to anomaly detection.

**Returns:**
- `pd.DataFrame`: Feature importance scores (feature name, importance)

**Example:**
```python
importance = scorer.get_feature_importance()
print(importance.head(5))
```

**Output:**
```
              feature  importance
0         total_spend       0.35
1   mcc_concentration       0.28
2    transaction_count       0.15
3      mcc_diversity       0.12
4     avg_ticket_size       0.10
```

---

#### `rank_anomalies(scores: pd.Series, top_n: int = 20) -> pd.DataFrame`

Rank and select top anomalies.

**Parameters:**
- `scores` (pd.Series): Anomaly scores
- `top_n` (int): Number of top anomalies to return

**Returns:**
- `pd.DataFrame`: Ranked anomalies with scores

**Example:**
```python
top_anomalies = scorer.rank_anomalies(scores, top_n=20)
```

---

## Reporting Layer API

### `src.reporting.generator.ReportGenerator`

Generates CSV and HTML reports from anomaly scores.

#### Constructor

```python
ReportGenerator(config: Dict[str, Any])
```

**Parameters:**
- `config` (dict): Reporting configuration from `modelconfig.yaml`

---

#### `generate_csv(anomalies: pd.DataFrame, output_path: str) -> None`

Export anomalies to CSV file.

**Parameters:**
- `anomalies` (pd.DataFrame): Ranked anomalies with features
- `output_path` (str): Output file path

**Raises:**
- `IOError`: If file cannot be written

**Example:**
```python
from src.reporting.generator import ReportGenerator

generator = ReportGenerator(config)
generator.generate_csv(anomalies, "outputs/anomalies_2025-11-21.csv")
```

---

#### `generate_html(anomalies: pd.DataFrame, output_path: str) -> None`

Create HTML dashboard with visualizations.

**Parameters:**
- `anomalies` (pd.DataFrame): Ranked anomalies with features
- `output_path` (str): Output file path

**Example:**
```python
generator.generate_html(anomalies, "outputs/report_2025-11-21.html")
```

**HTML Components:**
- Summary statistics (total anomalies, score distribution)
- Top 10 anomalies table
- Feature importance bar chart
- Opportunity vs. concern pie chart

---

#### `add_business_tags(anomalies: pd.DataFrame) -> pd.DataFrame`

Label anomalies as opportunities or concerns.

**Parameters:**
- `anomalies` (pd.DataFrame): Anomalies with feature values

**Returns:**
- `pd.DataFrame`: Anomalies with `opportunity_flag` and `concern_flag` columns

**Tagging Logic:**
```python
# Opportunity: high spend OR high diversity
if total_spend > 90th percentile OR mcc_diversity > 10:
    opportunity_flag = "high_spend" or "diversification"

# Concern: high concentration
if mcc_concentration > 0.8:
    concern_flag = "concentration_risk"
```

**Example:**
```python
tagged_anomalies = generator.add_business_tags(anomalies)
```

---

## Utilities API

### `src.utils.config_loader.ConfigLoader`

Loads and validates YAML configuration files.

#### `load(path: str) -> Dict[str, Any]`

Parse YAML configuration file.

**Parameters:**
- `path` (str): Path to config file

**Returns:**
- `dict`: Parsed configuration

**Raises:**
- `FileNotFoundError`: If config file missing
- `ConfigurationError`: If YAML invalid

**Example:**
```python
from src.utils.config_loader import ConfigLoader

loader = ConfigLoader()
config = loader.load("config/modelconfig.yaml")
```

---

### `src.utils.logger.StructuredLogger`

Provides JSON and text logging with context.

#### `get_logger(name: str) -> StructuredLogger`

Create or retrieve a logger instance.

**Parameters:**
- `name` (str): Logger name (typically module name)

**Returns:**
- `StructuredLogger`: Logger instance

**Example:**
```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing started", context={"customer_count": 10245})
```

---

#### `info(message: str, context: Dict = None) -> None`

Log informational message.

**Parameters:**
- `message` (str): Log message
- `context` (dict, optional): Additional structured data

**Example:**
```python
logger.info("Data loaded", context={"rows": 10245, "duration": 2.3})
```

---

#### `error(message: str, exception: Exception = None) -> None`

Log error with optional exception details.

**Parameters:**
- `message` (str): Error message
- `exception` (Exception, optional): Exception object for stacktrace

**Example:**
```python
try:
    data = loader.load()
except FileNotFoundError as e:
    logger.error("Failed to load data", exception=e)
    raise
```

---

### `src.utils.exceptions`

Custom exception classes for domain-specific errors.

#### Exception Classes

```python
class DataValidationError(Exception):
    """Raised when data fails validation checks"""
    pass

class SchemaValidationError(DataValidationError):
    """Raised when data schema is invalid"""
    pass

class BusinessRuleViolationError(DataValidationError):
    """Raised when business rules are violated"""
    pass

class ConfigurationError(Exception):
    """Raised when config files are invalid or missing"""
    pass

class ModelError(Exception):
    """Raised when model training or prediction fails"""
    pass
```

**Example:**
```python
from src.utils.exceptions import SchemaValidationError

def validate_schema(df):
    if "customer_id" not in df.columns:
        raise SchemaValidationError("Missing required column: customer_id")
```

---

## CLI Reference

### `cli.py` Commands

Command-line interface for system operations.

#### `validate` - Validate Data

```bash
python cli.py validate --data-config config/dataconfig.yaml
```

**Options:**
- `--data-config PATH`: Path to data configuration file (required)

**Output:**
- Prints validation results to console
- Exit code 0 if valid, 1 if errors

**Example:**
```bash
$ python cli.py validate --data-config config/dataconfig.yaml
✓ Schema validation passed
✓ Business rules validation passed
Data is valid for processing
```

---

#### `detect` - Run Anomaly Detection

```bash
python cli.py detect --model-config config/modelconfig.yaml --data-config config/dataconfig.yaml
```

**Options:**
- `--model-config PATH`: Path to model configuration file (required)
- `--data-config PATH`: Path to data configuration file (required)
- `--output-dir PATH`: Output directory for results (default: `outputs/`)

**Output:**
- CSV and HTML reports in output directory
- Logs to console and `logs/application.log`

---

#### `report` - Generate Report from Existing Results

```bash
python cli.py report --input outputs/anomalies.csv --output-dir outputs/
```

**Options:**
- `--input PATH`: Path to anomalies CSV file (required)
- `--output-dir PATH`: Output directory for HTML report (required)
- `--config PATH`: Optional reporting config (uses defaults if not provided)

---

### `main.py` Entry Point

Main pipeline execution script.

```bash
python main.py --model-config config/modelconfig.yaml --data-config config/dataconfig.yaml
```

**Options:**
- `--model-config PATH`: Path to model configuration (required)
- `--data-config PATH`: Path to data configuration (required)
- `--output-dir PATH`: Output directory (default: `outputs/`)
- `--log-level LEVEL`: Logging level (default: `INFO`)

**Example:**
```bash
python main.py \
  --model-config config/modelconfig.yaml \
  --data-config config/dataconfig.yaml \
  --output-dir outputs/ \
  --log-level DEBUG
```

---

## Configuration Schema

### Model Configuration (`modelconfig.yaml`)

```yaml
model:
  algorithm: str                 # "IsolationForest" (only supported algorithm in v1.0)
  contamination: float           # Expected anomaly percentage (0.01-0.10)
  n_estimators: int              # Number of trees (50-200 recommended)
  max_samples: int or str        # Samples per tree (256 or "auto")
  random_state: int              # Seed for reproducibility

features:
  use:                           # List of features to include
    - str                        # Feature names
  normalize: bool                # Whether to normalize features
  normalization_method: str      # "standard" or "minmax"

reporting:
  top_n: int                     # Number of anomalies to report
  output_formats:                # List of output formats
    - str                        # "csv", "html"
  business_rules:
    opportunity_thresholds:
      high_spend_percentile: float     # 0-100
      high_diversity_min: int          # Minimum MCC count
    concern_thresholds:
      concentration_risk_threshold: float  # 0-1 (Herfindahl index)
```

### Data Configuration (`dataconfig.yaml`)

```yaml
data:
  source:
    type: str                    # "csv", "parquet", "postgres", "s3"
    path: str                    # File path or connection string
    
  schema:
    required_columns:            # List of required columns
      - str
    types:                       # Column name -> data type mapping
      column_name: str           # "string", "float", "int"
  
  filters:
    min_transaction_count: int   # Minimum transactions per customer
    max_days_ago: int            # How far back to look
    reporting_week: str          # "latest" or ISO week "YYYY-Www"
  
  validation:
    allow_negatives: bool        # Whether to allow negative amounts
    max_spend_per_transaction: float  # Maximum reasonable transaction
```

---

## Code Examples

### Complete Pipeline Example

```python
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.features.builder import FeatureBuilder
from src.models.scorer import AnomalyScorer
from src.reporting.generator import ReportGenerator
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

# Setup
logger = get_logger(__name__)
config_loader = ConfigLoader()

# Load configurations
model_config = config_loader.load("config/modelconfig.yaml")
data_config = config_loader.load("config/dataconfig.yaml")

# 1. Load data
logger.info("Loading data")
loader = DataLoader(data_config)
df = loader.load()

# 2. Validate data
logger.info("Validating data")
validator = DataValidator(data_config)
validator.validate_schema(df)
validator.validate_business_rules(df)

# 3. Build features
logger.info("Building features")
builder = FeatureBuilder(model_config)
features = builder.build_customer_features(df)
features = builder.add_derived_features(features)
features = builder.normalize_features(features)

# 4. Train model and score
logger.info("Training model")
scorer = AnomalyScorer(model_config)
scorer.train(features)

logger.info("Scoring customers")
scores = scorer.score(features)
anomalies = scorer.rank_anomalies(scores, top_n=20)

# 5. Generate reports
logger.info("Generating reports")
generator = ReportGenerator(model_config)
anomalies = generator.add_business_tags(anomalies)
generator.generate_csv(anomalies, "outputs/anomalies.csv")
generator.generate_html(anomalies, "outputs/report.html")

logger.info("Pipeline completed successfully")
```

---

### Custom Feature Engineering Example

```python
from src.features.builder import FeatureBuilder
import pandas as pd

# Extend FeatureBuilder for custom features
class CustomFeatureBuilder(FeatureBuilder):
    def add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add week-over-week growth features"""
        # Calculate growth rates
        df['spend_growth_rate'] = df.groupby('customer_id')['total_spend'].pct_change()
        df['transaction_growth_rate'] = df.groupby('customer_id')['transaction_count'].pct_change()
        return df

# Use custom builder
builder = CustomFeatureBuilder(config)
features = builder.build_customer_features(df)
features = builder.add_velocity_features(features)
```

---

### Custom Business Rules Example

```python
from src.reporting.generator import ReportGenerator

class CustomReportGenerator(ReportGenerator):
    def add_industry_tags(self, anomalies: pd.DataFrame) -> pd.DataFrame:
        """Tag anomalies by industry vertical"""
        # Map MCCs to industries
        industry_map = {
            '5411': 'Grocery',
            '5812': 'Restaurants',
            '5541': 'Gas Stations'
        }
        
        anomalies['top_industry'] = anomalies['top_mcc'].map(industry_map)
        return anomalies

# Use custom generator
generator = CustomReportGenerator(config)
anomalies = generator.add_business_tags(anomalies)
anomalies = generator.add_industry_tags(anomalies)
```

---

## Versioning & Compatibility

### API Stability

- **Stable (v1.0)**: All public methods documented above
- **Experimental**: Methods prefixed with `_experimental_`
- **Internal**: Methods prefixed with `_` (not documented, may change)

### Deprecation Policy

Deprecated methods will:
1. Emit `DeprecationWarning` for 2 minor versions
2. Be removed in next major version

**Example:**
```python
import warnings
warnings.warn("method_name() is deprecated, use new_method() instead", DeprecationWarning)
```

---

## Support & Contribution

### Reporting Issues

Open issues on GitHub: [Project Repository](https://github.com/dscv101/card-portfolio-anomaly-detection/issues)

### Contributing

See [AGENTS.md](../AGENTS.md) for coding standards and contribution guidelines.

---

**Document Maintained By**: Data Science Team  
**Last Review**: 2025-11-21  
**Next Review**: 2025-12-21

