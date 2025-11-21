# Production Deployment Guide

Complete guide for deploying the Card Portfolio Anomaly Detection System to production.

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Ubuntu 20.04 LTS or later |
| Python | 3.9+ |
| Memory | 16GB minimum |
| Storage | 100GB SSD |

## Prerequisites

1. Server access with sudo privileges
2. Database credentials
3. SMTP configuration for alerts
4. Airflow admin access

## Server Setup

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install python3.9 python3-pip -y

# Clone repository
git clone https://github.com/bank/anomaly-detection.git
cd anomaly-detection

# Install requirements
pip3 install -r requirements.txt
pip3 install -r requirements-airflow.txt
```

## Configuration

Create `.env` file with:
```
DB_CONNECTION_STRING=postgresql://user:pass@host:5432/db
SMTP_HOST=smtp.bank.com
SMTP_PORT=587
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

## Airflow Setup

```bash
# Initialize Airflow database
airflow db init

# Create admin user
airflow users create --username admin --password <password> \
  --firstname Admin --lastname User --email admin@bank.com --role Admin

# Start Airflow
airflow webserver -p 8080 &
airflow scheduler &
```

## Monitoring Setup

Add cron job for health checks:
```bash
0 */4 * * * /usr/bin/python3 /opt/anomaly-detection/monitoring/health_check.py
```

## Power BI Integration

1. Open Power BI Desktop
2. Get Data → Database → PostgreSQL
3. Enter credentials from `.env`
4. Select `anomaly_reports` table
5. Publish to Power BI Service
6. Configure scheduled refresh (Daily 8AM CST)

## Deployment Checklist

### Pre-Deployment
- [ ] Database schema created
- [ ] Environment variables configured
- [ ] Dependencies installed
- [ ] Tests passing

### During Deployment
- [ ] Code deployed to production server
- [ ] Airflow DAG deployed and visible
- [ ] Monitoring cron job active
- [ ] Power BI dashboard connected

### Post-Deployment
- [ ] Manual test run successful
- [ ] Alerts configured and tested
- [ ] Dashboard refreshing correctly
- [ ] Documentation updated

## Runbook

### Daily Operations
1. Check Airflow UI for successful runs
2. Review anomaly reports in Power BI
3. Monitor alert channels

### Troubleshooting

**Issue: Airflow DAG not running**
- Check scheduler status: `ps aux | grep airflow`
- Review logs: `tail -f ~/airflow/logs/scheduler/latest/*.log`

**Issue: No anomalies detected**
- Verify input data freshness
- Check model files exist in `models/`
- Review pipeline logs

**Issue: Email alerts not sending**
- Test with: `python -c "import smtplib; s=smtplib.SMTP('smtp.bank.com', 587); s.quit()"`
- Test with: `python -m smtplib`

## Rollback Procedure

1. Stop Airflow scheduler
2. Revert to previous Git tag: `git checkout v1.x.x`
3. Restart services
4. Verify functionality

## Support

For production issues, contact:
- DevOps: devops@bank.com
- Data Science: data-science-team@bank.com
