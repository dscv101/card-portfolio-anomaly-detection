# Sample Data for E2E Testing

This directory contains sample transaction data files used for end-to-end testing of the anomaly detection pipeline.

## Data Schema

All data files follow this schema:

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | Unique customer identifier (e.g., "CUST00001") |
| `reporting_week` | string | ISO week start date (YYYY-MM-DD format) |
| `mcc` | string | Merchant Category Code (4 digits) |
| `spend_amount` | float | Total spend amount for the customer-MCC-week |
| `transaction_count` | int | Number of transactions |
| `avg_ticket_amount` | float | Average transaction amount (spend/count) |

## File Formats

Files are available in two formats:

- **CSV**: `transactions_{reporting_week}.csv`
- **Parquet**: `transactions_{reporting_week}.parquet`

Both formats contain identical data and can be used interchangeably for testing.

## Data Characteristics

### Normal Customers (CUST00001 - CUST00500)

Generated with realistic spending patterns:

- **70% Low Spenders**: 1-20 transactions, $50-$1,000 total spend
- **15% Medium Spenders**: 10-50 transactions, $500-$3,000 total spend
- **15% High Spenders**: 20-100 transactions, $2,000-$10,000 total spend

Each customer has 1-6 active MCC codes representing diverse spending categories.

### Edge Case Customers (EDGE001 - EDGE007)

Special customers designed to trigger specific anomaly patterns:

| Customer | Pattern | Description |
|----------|---------|-------------|
| `EDGE001` | High Growth | Large spend increase (potential high_growth_opportunity tag) |
| `EDGE002` | High Concentration | 95% of spend in single MCC (concentration_concern tag) |
| `EDGE003` | Unusual Ticket | Very high average ticket amount (unusual_ticket_size tag) |
| `EDGE004` | Single MCC | Only one active merchant category |
| `EDGE005` | New High Spender | High spend customer with limited history (new_high_spender tag) |
| `EDGE006` | Zero Spend | Boundary case with no transactions |
| `EDGE007` | Many Small Txns | 200 transactions averaging $2.50 each |

### MCC Codes

The data includes 8 common merchant category codes:

| MCC | Category |
|-----|----------|
| 5411 | Grocery Stores |
| 5812 | Restaurants |
| 5541 | Gas Stations |
| 5999 | Misc Retail |
| 5732 | Electronics |
| 5912 | Drug Stores |
| 5311 | Department Stores |
| 5814 | Fast Food |

## Historical Data

File: `transactions_historical.parquet`

Contains 12 weeks of historical data for ~80% of customers (randomly sampled per week). Used for computing delta features like:

- `spend_growth_4w`: 4-week growth rate
- `spend_growth_12w`: 12-week growth rate
- Historical spend patterns

## Data Generation

Data is generated using `scripts/generate_sample_data.py` with a fixed random seed (42) for reproducibility.

### Generate Current Week Data Only

```bash
python scripts/generate_sample_data.py --reporting-week 2025-11-18
```

### Generate with Historical Data

```bash
python scripts/generate_sample_data.py --reporting-week 2025-11-18 --with-historical
```

### Custom Parameters

```bash
python scripts/generate_sample_data.py \
    --n-customers 1000 \
    --reporting-week 2025-11-25 \
    --output-dir ./datasamples \
    --seed 12345 \
    --with-historical
```

## Data Volumes

For default parameters (500 customers):

- **Current week file**: ~1,500 transactions, ~100 KB (CSV), ~50 KB (Parquet)
- **Historical file**: ~30,000 transactions, ~2 MB (CSV), ~1 MB (Parquet)

## Testing Usage

### Unit Tests

Load specific data files for targeted testing:

```python
from src.data.loader import DataLoader

loader = DataLoader(config)
data = loader.load("2025-11-18")  # Loads transactions_2025-11-18.csv or .parquet
```

### E2E Tests

Full pipeline testing with sample data:

```python
from main import run_anomaly_detection

result = run_anomaly_detection(
    reporting_week="2025-11-18",
    mode="adhoc"
)
assert result["status"] == "success"
```

### Reproducibility Tests

The fixed random seed ensures deterministic data generation:

```python
# Run 1
python scripts/generate_sample_data.py --seed 42
data1 = pd.read_csv("datasamples/transactions_2025-11-18.csv")

# Run 2
python scripts/generate_sample_data.py --seed 42
data2 = pd.read_csv("datasamples/transactions_2025-11-18.csv")

# Data should be identical
assert data1.equals(data2)
```

## Maintenance

When updating sample data:

1. Modify `scripts/generate_sample_data.py` as needed
2. Regenerate files with consistent seed
3. Update this README if schema or characteristics change
4. Commit both script and generated data files

## Notes

- Data is synthetic and does not represent real customer information
- All customer IDs and patterns are randomly generated
- Use this data only for development and testing purposes
- For production testing, use anonymized production data samples

