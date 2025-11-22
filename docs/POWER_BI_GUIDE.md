# Power BI Integration Guide

This guide provides instructions for connecting Power BI to the Card Portfolio Anomaly Detection System.

## Data Source Configuration

### Option 1: File-Based (Development)
Connect to CSV files in the `outputs/` directory.

### Option 2: Database (Production - Recommended)
Connect to the `anomaly_reports` table.

## Dashboard Design

### KPI Cards
- Total Anomalies This Week
- Critical Risk Count  
- Average Anomaly Score

### Visuals
- Anomaly Trend Line (13 weeks)
- Risk Category Distribution
- Top 10 Anomalies Table

## Refresh Schedule
Configure scheduled refresh in Power BI Service:
- Daily at 8:00 AM CST
- Enable refresh failure notifications

## Support
For issues, contact bi-team@bank.com
