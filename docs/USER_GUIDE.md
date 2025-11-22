# User Guide: Card Portfolio Anomaly Detection System

**Version:** 1.0.0  
**Last Updated:** 2025-11-21  
**Target Audience:** Business Analysts, Product Owners, Data Analysts

---

## Overview

The Card Portfolio Anomaly Detection System is an automated tool that identifies unusual patterns in credit card customer behavior. It helps you discover high-value opportunities (like rapidly growing customers) and potential concerns (like concentration risks) without manual data analysis.

### What This System Does

- **Detects Anomalies**: Automatically identifies the top 20 most unusual customers each week
- **Provides Context**: Shows why each customer is flagged (spending patterns, transaction behavior, etc.)
- **Enables Action**: Generates clear reports for business review and decision-making

### What This System Does NOT Do

- ❌ Does not perform real-time fraud detection
- ❌ Does not automatically contact customers or trigger account actions
- ❌ Does not track individual customer behavior over time (focuses on cross-sectional comparisons)

---

## Getting Started

### System Access

**Prerequisites:**
- Access to the production data warehouse
- Python 3.9+ installed on your system
- Permissions to read customer transaction data

**Initial Setup:**

1. Clone the repository:

   ```bash
   git clone https://github.com/your-company/card-portfolio-anomaly-detection.git
   cd card-portfolio-anomaly-detection
   ```

2. Install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Verify installation:

   ```bash
   python -c "import pandas, sklearn; print('Setup complete!')"
   ```

---

## Running the System

### Basic Usage

**1. Run a Weekly Analysis:**

```bash
python main.py --model-config config/modelconfig.yaml --data-config config/dataconfig.yaml
```

This command will:
- Load customer transaction data for the most recent reporting week
- Calculate behavioral features (spending patterns, transaction counts, etc.)
- Run anomaly detection using the Isolation Forest algorithm
- Generate a report with the top 20 anomalous customers
- Save results to `outputs/anomalies_YYYY-MM-DD.csv` and `outputs/report_YYYY-MM-DD.html`

**2. Use the CLI for Specific Tasks:**

```bash
# Validate your data before analysis
python cli.py validate --data-config config/dataconfig.yaml

# Run anomaly detection on existing data
python cli.py detect --model-config config/modelconfig.yaml

# Generate a custom report
python cli.py report --input outputs/anomalies.csv --output-dir outputs/
```

---

## Understanding the Output

### Anomaly Report Structure

The system generates two main outputs:

#### 1. CSV Report (`anomalies_YYYY-MM-DD.csv`)

Contains one row per anomalous customer with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `customer_id` | Unique customer identifier | `CUST_12345` |
| `anomaly_score` | Anomaly strength (approximately -0.5 to 0.5, lower = more anomalous) | `-0.25` |
| `anomaly_rank` | Rank among detected anomalies (1 = most anomalous) | `1` |
| `total_spend` | Total spending for the reporting week | `15000.00` |
| `transaction_count` | Number of transactions | `45` |
| `avg_ticket_size` | Average transaction amount | `333.33` |
| `mcc_diversity` | Number of different merchant categories | `12` |
| `mcc_concentration` | Spend concentration in top category | `0.45` |
| `opportunity_flag` | Indicates positive outlier | `high_spend` |
| `concern_flag` | Indicates risk pattern | `concentration_risk` |

#### 2. HTML Report (`report_YYYY-MM-DD.html`)

A visual dashboard with:
- Summary statistics (total anomalies, average scores)
- Top 10 customers by anomaly strength
- Feature importance charts (which behaviors drove the detection)
- Opportunity vs. concern breakdown

**Accessing the Report:**

Open the HTML file in any web browser:
```bash
# On Mac/Linux
open outputs/report_YYYY-MM-DD.html

# On Windows
start outputs/report_YYYY-MM-DD.html
```

---

## Interpreting Results

### What Makes a Customer Anomalous?

The system looks for unusual combinations of features:

**Positive Outliers (Opportunities):**
- 🟢 **High Spend**: Customers spending significantly more than peers
- 🟢 **Growth**: Large increases in transaction volume
- 🟢 **Diversification**: Spending across many different merchant categories

**Concern Signals:**
- 🔴 **Concentration Risk**: High spending in a single merchant category
- 🔴 **Unusual Patterns**: Behaviors that don't match typical customer profiles

### Anomaly Score Interpretation

The anomaly score typically ranges from approximately **-0.5 to 0.5**:
- **Negative scores** indicate outliers (more anomalous behavior) - these customers differ significantly from typical patterns
- **Positive scores** indicate inliers (less anomalous behavior) - these customers behave similarly to the majority

**Example interpretation:**
- A customer with score **-0.35** is highly anomalous (e.g., spending $50,000/week when typical is $5,000)
- A customer with score **-0.15** is moderately anomalous (e.g., 3x normal transaction volume)
- A customer with score **0.10** is very typical (normal spending and transaction patterns)

**Action thresholds:**
- **Score < -0.2**: Strong anomaly (review immediately)
- **Score -0.1 to -0.2**: Moderate anomaly (investigate when possible)
- **Score > -0.1**: Weak anomaly (likely normal behavior with minor variations)

### Feature Importance

The report shows which features contributed most to each anomaly:

```text
Top Features for CUST_12345:
1. total_spend: 98th percentile (+$12,000 above median)
2. mcc_diversity: 15 categories (typical: 5-8)
3. avg_ticket_size: $500 (typical: $50-150)
```

---

## Review Workflow

### Step 1: Triage Anomalies

Review the top 20 anomalies and categorize them:

1. **Actionable Opportunities**: Customers worth engaging (e.g., high spenders)
2. **Monitor**: Customers showing concerning patterns (e.g., concentration risk)
3. **False Positives**: Anomalies explained by known factors (e.g., seasonal purchases)

### Step 2: Document Findings

Use the provided template:

```
Customer ID: CUST_12345
Anomaly Type: High Spend + Diversification
Business Context: New merchant partnership program participant
Action: Add to high-value customer outreach list
Owner: Account Management Team
Due Date: 2025-12-01
```

### Step 3: Take Action

Based on your findings:
- **For Opportunities**: Share with sales/account management for outreach
- **For Concerns**: Escalate to risk management or compliance
- **For False Positives**: Update documentation for future reference

### Step 4: Provide Feedback

Help improve the system by tracking outcomes:
- What % of flagged anomalies were meaningful?
- What actions were taken?
- Were there any missed anomalies (customers you expected to see but didn't)?

Share feedback with the Data Science Team via the project repository or Slack channel.

---

## Configuration Options

### Adjusting Detection Sensitivity

Edit `config/modelconfig.yaml`:

```yaml
model:
  contamination: 0.05  # Percentage of data expected to be anomalous (default: 5%)
  n_estimators: 100    # Number of decision trees (higher = more stable, slower)
  
reporting:
  top_n: 20            # Number of anomalies to report (default: 20)
```

**Common Adjustments:**
- Increase `contamination` to 0.10 if you want more anomalies detected
- Decrease `contamination` to 0.03 if you want only the most extreme cases
- Increase `top_n` to 30 if you need more candidates for review

### Data Source Configuration

Edit `config/dataconfig.yaml`:

```yaml
data:
  source:
    path: "s3://bucket/data/transactions.csv"  # Update to your data location
    format: "csv"
  
  filters:
    min_transaction_count: 5  # Exclude low-activity customers
    reporting_week: "latest"  # Or specify: "2025-W47"
```

---

## Troubleshooting

### Common Issues

**Problem:** "No anomalies detected"

**Solution:** 
- Check if data loaded correctly: `python cli.py validate --data-config config/dataconfig.yaml`
- Increase `contamination` parameter in `modelconfig.yaml`
- Verify that the data has enough variation (not all customers behaving identically)

---

**Problem:** "Too many false positives"

**Solution:**
- Decrease `contamination` parameter to focus on stronger anomalies
- Add data filters in `dataconfig.yaml` to exclude known edge cases
- Review feature engineering settings to reduce noise

---

**Problem:** "Report generation failed"

**Solution:**
- Check that output directory exists: `mkdir -p outputs/`
- Verify write permissions: `ls -la outputs/`
- Review logs in `logs/application.log` for specific errors

---

**Problem:** "Data validation errors"

**Solution:**
- Check data schema matches expected format (see `dataconfig.yaml`)
- Ensure all required columns are present: `customer_id`, `reporting_week`, `mcc`, `spend_amount`, `transaction_count`
- Verify data types (numeric fields should not contain text)

---

## Best Practices

### Weekly Execution

Run the system every Monday to analyze the previous week:

```bash
# Set up a cron job (Linux/Mac)
0 9 * * 1 cd /path/to/project && ./venv/bin/python main.py --model-config config/modelconfig.yaml --data-config config/dataconfig.yaml
```

### Data Quality Checks

Always run validation before detection:

```bash
python cli.py validate --data-config config/dataconfig.yaml
```

### Version Control

Keep track of configuration changes:
- Document why you changed sensitivity parameters
- Save old reports before re-running with new settings
- Track which anomalies led to business actions

### Collaboration

Share findings effectively:
- Export CSV results to your BI tool for deeper analysis
- Include context in emails (don't just send raw anomaly scores)
- Schedule regular reviews with stakeholders to discuss patterns

---

## Frequently Asked Questions

**Q: How often should I run this system?**

A: Weekly is recommended for most portfolios. Monthly may work for slower-moving businesses.

---

**Q: Can I analyze a specific time period?**

A: Yes, edit `dataconfig.yaml` and set `reporting_week: "2025-W45"` (ISO week format).

---

**Q: What if I want to detect different types of anomalies?**

A: Contact the Data Science Team. Feature engineering can be customized to focus on specific behaviors (e.g., geographic patterns, time-of-day anomalies).

---

**Q: Is this system suitable for real-time monitoring?**

A: No, this is a batch system optimized for weekly/monthly retrospective analysis. For real-time needs, consider a different architecture.

---

**Q: How do I export results to Excel?**

A: The CSV output can be opened directly in Excel:
```bash
open outputs/anomalies_2025-11-21.csv
```

Or convert programmatically:
```python
import pandas as pd
df = pd.read_csv('outputs/anomalies_2025-11-21.csv')
df.to_excel('outputs/anomalies_2025-11-21.xlsx', index=False)
```

---

**Q: Who do I contact for support?**

A: Reach out to the Data Science Team via:
- Slack: #data-science-support
- Email: datascience@company.com
- GitHub Issues: [Project Repository](https://github.com/dscv101/card-portfolio-anomaly-detection/issues)

---

## Appendix: Sample Output

### Example Anomaly Record

```csv
customer_id,anomaly_score,anomaly_rank,total_spend,transaction_count,avg_ticket_size,mcc_diversity,mcc_concentration,opportunity_flag,concern_flag
CUST_98765,-0.31,1,48500.00,127,381.89,18,0.28,high_spend+diversification,
CUST_45612,-0.28,2,8900.00,12,741.67,3,0.89,,concentration_risk
CUST_77234,-0.25,3,35000.00,89,393.26,14,0.35,high_spend,
```

### Interpretation:

1. **CUST_98765** (Rank 1): High-spending, diversified customer - excellent opportunity
2. **CUST_45612** (Rank 2): Low diversity, high concentration - potential concern
3. **CUST_77234** (Rank 3): High-spending with moderate diversity - opportunity

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-21 | Initial user guide release |

---

**Need Help?** Contact the Data Science Team or open an issue on GitHub.
