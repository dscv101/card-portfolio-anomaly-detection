# Power BI Integration Guide

**Version:** 1.0.0  
**Status:** Production  
**Owner:** BI Developer Team  
**Authority Level:** Platform (REQ-7.2)

## Overview

This guide provides instructions for connecting Power BI to the Card Portfolio Anomaly Detection System and building the executive dashboard.

## Data Source Configuration

### Option 1: File-Based (Development)

Connect to CSV files in the `outputs/` directory:

```powerquery
let
    Source = Folder.Files("C:\path\to\outputs"),
    FilteredFiles = Table.SelectRows(Source, each Text.StartsWith([Name], "anomalies_")),
    CombinedFiles = Table.Combine(FilteredFiles[Content])
in
    CombinedFiles
```

### Option 2: Database (Production - Recommended)

Connect to the `anomaly_reports` table in the production database:

**SQL Schema:**
```sql
CREATE TABLE anomaly_reports (
    customer_id VARCHAR(50) NOT NULL,
    reporting_week VARCHAR(7) NOT NULL,
    anomaly_score FLOAT NOT NULL,
    anomaly_rank INT NOT NULL,
    risk_category VARCHAR(20),
    total_spend DECIMAL(12, 2),
    transaction_count INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id, reporting_week)
);

CREATE INDEX idx_reporting_week ON anomaly_reports(reporting_week);
CREATE INDEX idx_anomaly_rank ON anomaly_reports(anomaly_rank);
```

**Power BI Connection:**
1. Get Data → SQL Server
2. Server: `your-db-server.database.windows.net`
3. Database: `card_analytics`
4. Table: `anomaly_reports`

## Dashboard Design

### Page 1: Executive Summary

**KPI Cards:**
- Total Anomalies This Week
- Critical Risk Count
- Average Anomaly Score
- YoY Change %

**Visuals:**
- Anomaly Trend Line (13 weeks)
- Risk Category Distribution (Pie)
- Top 10 Anomalies Table

### Page 2: Detailed Analysis

**Visuals:**
- Scatter Plot: Spend vs Anomaly Score
- Heatmap: Risk by Week
- Table: All Anomalies with filters

### Page 3: Customer Drill-Down

**Interactive table with drill-through:**
- Customer ID
- Reporting Week
- Anomaly Details
- Transaction History

## Refresh Schedule

Configure scheduled refresh in Power BI Service:

1. Navigate to Workspace → Dataset Settings
2. Schedule refresh: **Daily at 8:00 AM CST**
3. Enable: Refresh failure notifications
4. Email: bi-team@bank.com

## Sharing and Permissions

1. Publish to workspace: `Card Analytics`
2. Create App: `Anomaly Detection Dashboard`
3. Share with:
   - Executive Team (View)
   - Risk Management (View + Filter)
   - Data Science Team (Edit)

## Troubleshooting

### Dashboard Not Refreshing

1. Check data source credentials
2. Verify gateway connection (if on-premises)
3. Review refresh history for errors
4. Ensure data files exist for current week

### Missing Data

1. Verify pipeline executed successfully
2. Check output files in `outputs/` directory
3. Validate database table has latest records
4. Review Airflow DAG execution logs

## Support

For issues, contact:
- BI Team: bi-team@bank.com
- Data Science Team: data-science-team@bank.com

