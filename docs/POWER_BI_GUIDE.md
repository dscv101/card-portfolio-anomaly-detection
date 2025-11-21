# Power BI Integration Guide

**Version:** 1.0.0  
**Status:** Production  
**Owner:** BI Development Team  
**Date:** 2025-11-21  
**Authority Level:** Platform (REQ-7.2)

---

## Overview

This guide describes how to connect Power BI to the anomaly detection system outputs and build dashboards for stakeholder consumption.

### Purpose

Enable business stakeholders to:
- View the latest anomaly detection results
- Track anomaly trends over time
- Drill down by customer, category, and risk level
- Export reports for decision-making

---

## 1. Data Source Configuration

### 1.1 Output Data Structure

The anomaly detection pipeline produces the following output files in `outputs/`:

```
outputs/
├── anomalies_YYYY-WW.csv       # Main anomaly report
├── features_YYYY-WW.csv        # Feature values for all customers
└── metadata_YYYY-WW.json       # Execution metadata
```

### 1.2 Power BI Data Source Setup

#### Option A: Direct File Connection (Development)

1. Open Power BI Desktop
2. Click **Get Data** → **Text/CSV**
3. Navigate to the `outputs/` directory
4. Select `anomalies_[latest].csv`
5. Click **Transform Data** to open Power Query Editor

#### Option B: Database Connection (Production)

For production deployments, load the CSV outputs into a database table:

**Recommended Schema:**

```sql
CREATE TABLE anomaly_reports (
    customer_id VARCHAR(50) NOT NULL,
    reporting_week VARCHAR(7) NOT NULL,
    anomaly_score FLOAT NOT NULL,
    anomaly_rank INT NOT NULL,
    risk_category VARCHAR(20),
    total_spend DECIMAL(12, 2),
    transaction_count INT,
    mcc_diversity INT,
    top_mcc VARCHAR(50),
    top_mcc_spend DECIMAL(12, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id, reporting_week)
);

CREATE INDEX idx_reporting_week ON anomaly_reports(reporting_week);
CREATE INDEX idx_anomaly_rank ON anomaly_reports(anomaly_rank);
CREATE INDEX idx_risk_category ON anomaly_reports(risk_category);
```

**Connection String Example:**

```
Server=your-server.database.windows.net;
Database=card_analytics;
Trusted_Connection=False;
Encrypt=True;
TrustServerCertificate=False;
```

---

## 2. Power Query Transformations

### 2.1 Data Type Adjustments

In Power Query Editor, set the following data types:

| Column | Power BI Type |
|--------|---------------|
| customer_id | Text |
| reporting_week | Text |
| anomaly_score | Decimal Number |
| anomaly_rank | Whole Number |
| risk_category | Text |
| total_spend | Currency |
| transaction_count | Whole Number |
| mcc_diversity | Whole Number |
| top_mcc | Text |
| top_mcc_spend | Currency |

### 2.2 Calculated Columns

Add these calculated columns in Power Query or DAX:

**Reporting Week Date:**
```dax
ReportingWeekDate = 
    DATE(
        VALUE(LEFT([reporting_week], 4)),
        1, 
        1
    ) + ([reporting_week] - 1) * 7
```

**Risk Level:**
```dax
RiskLevel = 
    SWITCH(
        TRUE(),
        [anomaly_rank] <= 5, "Critical",
        [anomaly_rank] <= 10, "High",
        [anomaly_rank] <= 15, "Medium",
        "Low"
    )
```

**Spend Bucket:**
```dax
SpendBucket = 
    SWITCH(
        TRUE(),
        [total_spend] >= 100000, "$100K+",
        [total_spend] >= 50000, "$50K-$100K",
        [total_spend] >= 25000, "$25K-$50K",
        [total_spend] >= 10000, "$10K-$25K",
        "< $10K"
    )
```

---

## 3. Dashboard Design

### 3.1 Recommended Visuals

#### Page 1: Executive Summary

**Visual 1: KPI Cards**
- Total Anomalies Detected (this week)
- Average Anomaly Score
- Top Risk Category Count
- Week-over-Week Change %

**Visual 2: Anomaly Trend Line Chart**
```
X-Axis: Reporting Week Date
Y-Axis: Count of Anomalies
Legend: Risk Level
```

**Visual 3: Category Distribution (Donut Chart)**
```
Values: Count of Anomalies
Legend: Risk Category
```

**Visual 4: Top 10 Anomalies Table**
```
Columns:
- Rank
- Customer ID (masked: show last 4 digits)
- Anomaly Score
- Total Spend
- Risk Category
- Top MCC
```

#### Page 2: Detailed Analysis

**Visual 1: Scatter Plot**
```
X-Axis: Total Spend
Y-Axis: Anomaly Score
Size: Transaction Count
Legend: Risk Category
Tooltip: Customer ID, MCC Diversity
```

**Visual 2: Spend Distribution by Risk Level**
```
Chart Type: Stacked Bar
X-Axis: Risk Level
Y-Axis: Sum of Total Spend
Legend: Spend Bucket
```

**Visual 3: MCC Diversity vs Anomaly Rank**
```
Chart Type: Line and Clustered Column
Primary Axis: Anomaly Rank
Secondary Axis: Average MCC Diversity
```

#### Page 3: Customer Drill-Down

**Visual 1: Customer Search Slicer**
```
Type: Dropdown
Field: Customer ID
```

**Visual 2: Customer Anomaly History**
```
Chart Type: Line Chart
X-Axis: Reporting Week
Y-Axis: Anomaly Score
Filters: Selected Customer ID
```

**Visual 3: Customer Feature Details Table**
```
Rows: Feature Name
Values: Feature Value
Filters: Selected Customer ID, Latest Week
```

### 3.2 Color Scheme

Use consistent colors for risk levels:

| Risk Level | Color (Hex) | RGB |
|------------|-------------|-----|
| Critical | #D32F2F | 211, 47, 47 |
| High | #F57C00 | 245, 124, 0 |
| Medium | #FBC02D | 251, 192, 45 |
| Low | #388E3C | 56, 142, 60 |

---

## 4. Data Refresh Configuration

### 4.1 Scheduled Refresh (Power BI Service)

1. Publish report to Power BI Service
2. Navigate to **Settings** → **Datasets** → **[Your Dataset]**
3. Expand **Scheduled refresh**
4. Configure:
   - **Frequency:** Daily
   - **Time:** 8:00 AM CST (after Airflow DAG completes)
   - **Time zone:** Central Standard Time
   - **Send failure notifications:** data-science-team@bank.com

### 4.2 Gateway Configuration (if using database)

1. Install **Power BI Gateway** on a server with database access
2. Register the gateway in Power BI Service
3. Map the data source to the gateway
4. Configure credentials for database connection

---

## 5. Sharing and Permissions

### 5.1 Workspace Setup

Create a dedicated workspace:
- **Name:** Card Portfolio Analytics
- **License:** Power BI Pro or Premium
- **Members:**
  - Data Science Team (Admin)
  - Risk Management (Member)
  - Executive Leadership (Viewer)

### 5.2 Report Sharing

**Method 1: Workspace Access**
- Add users to the workspace with appropriate roles

**Method 2: App Publishing**
1. Create an **App** from the workspace
2. Customize app navigation and branding
3. Publish to organization
4. Share app link with stakeholders

**Method 3: Email Subscriptions**
1. Open report in Power BI Service
2. Click **Subscribe** → **Add new subscription**
3. Configure:
   - **Recipients:** stakeholder-email@bank.com
   - **Frequency:** Weekly (Tuesday morning)
   - **Include:** Snapshot of first page

---

## 6. Performance Optimization

### 6.1 Data Model Best Practices

- Use **Import** mode for small datasets (< 1M rows)
- Use **DirectQuery** for large datasets (> 1M rows)
- Create aggregations for summary tables
- Remove unused columns from queries
- Disable auto date/time hierarchy if not needed

### 6.2 Visual Optimization

- Limit visuals per page to 10-15
- Use **Top N** filters to reduce data volume
- Avoid high-cardinality fields in slicers
- Use **Aggregations** instead of row-level visuals

---

## 7. Troubleshooting

### Common Issues

#### Issue: Data not refreshing

**Solution:**
1. Check Airflow DAG execution logs
2. Verify output files exist in `outputs/` directory
3. Check Power BI refresh history for errors
4. Validate database connection credentials

#### Issue: Incorrect date parsing

**Solution:**
1. Ensure `reporting_week` format is consistent (YYYY-WW)
2. Check regional settings in Power BI
3. Use explicit date parsing in Power Query:
   ```m
   = Date.FromText(Text.Start([reporting_week], 4) & "-01-01")
   ```

#### Issue: Slow dashboard performance

**Solution:**
1. Reduce number of visuals on a single page
2. Apply filters to limit data volume
3. Use aggregated tables for summary views
4. Consider switching to DirectQuery mode

---

## 8. Maintenance

### 8.1 Weekly Review

Every Tuesday (after Monday pipeline run):
1. Verify data refresh completed successfully
2. Review anomaly counts for unusual spikes/drops
3. Check for data quality issues
4. Validate calculated measures

### 8.2 Monthly Audit

First week of each month:
1. Archive old output files (keep last 13 weeks)
2. Review dashboard usage metrics
3. Gather stakeholder feedback
4. Update documentation if needed

---

## 9. Support and Contacts

| Role | Contact | Responsibility |
|------|---------|----------------|
| BI Developer | bi-team@bank.com | Dashboard development, refresh issues |
| Data Science Team | data-science-team@bank.com | Data quality, model questions |
| DevOps | devops@bank.com | Airflow DAG, infrastructure |
| End Users | risk-management@bank.com | Business questions, feature requests |

---

## 10. References

- [Power BI Documentation](https://docs.microsoft.com/en-us/power-bi/)
- [DAX Function Reference](https://dax.guide/)
- [Power Query M Reference](https://docs.microsoft.com/en-us/powerquery-m/)
- [Anomaly Detection System Design](design.md)
- [Requirements Specification](REQUIREMENTS.md)

---

**Appendix A: Sample Power Query Script**

```m
let
    Source = Csv.Document(File.Contents("outputs/anomalies_latest.csv"),[Delimiter=",", Columns=10, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangedTypes = Table.TransformColumnTypes(PromotedHeaders,{
        {"customer_id", type text}, 
        {"reporting_week", type text}, 
        {"anomaly_score", type number}, 
        {"anomaly_rank", Int64.Type},
        {"risk_category", type text},
        {"total_spend", Currency.Type},
        {"transaction_count", Int64.Type},
        {"mcc_diversity", Int64.Type},
        {"top_mcc", type text},
        {"top_mcc_spend", Currency.Type}
    }),
    AddedWeekDate = Table.AddColumn(ChangedTypes, "ReportingWeekDate", each Date.StartOfWeek(Date.FromText(Text.Start([reporting_week], 4) & "-01-01"), Day.Monday))
in
    AddedWeekDate
```

---

**Document Change Log**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-21 | Codegen Bot | Initial version |

