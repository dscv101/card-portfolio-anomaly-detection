# Deployment Guide: Card Portfolio Anomaly Detection System

**Version:** 1.0.0  
**Authority Level:** Platform (REQ-7.4)

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Server Setup](#server-setup)
4. [Configuration](#configuration)
5. [Airflow Setup](#airflow-setup)
6. [Monitoring Setup](#monitoring-setup)
7. [Power BI Integration](#power-bi-integration)
8. [Deployment Checklist](#deployment-checklist)
9. [Runbook](#runbook)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This guide provides step-by-step instructions for deploying the Card Portfolio Anomaly Detection System to production.

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| Disk | 100 GB | 250 GB |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 LTS |
| Python | 3.9+ | 3.11+ |

---

## Prerequisites

**Software Dependencies:**
- Python 3.9 or higher
- Git 2.30+
- PostgreSQL 13+
- Apache Airflow 2.8+
- SMTP server access

**Access Requirements:**
- Database credentials
- SMTP credentials  
- Slack webhook URL (optional)
- SSH access to deployment server

---

## Server Setup

### Step 1: Create Deployment User

```bash
sudo useradd -m -s /bin/bash anomaly-detection
sudo usermod -aG sudo anomaly-detection
sudo su - anomaly-detection
```

### Step 2: Install System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-dev python3.11-venv python3-pip build-essential git curl libpq-dev
```

### Step 3: Clone Repository

```bash
mkdir -p /opt/anomaly-detection
cd /opt/anomaly-detection
git clone https://github.com/your-org/card-portfolio-anomaly-detection.git .
git checkout v1.0.0
```

### Step 4: Create Python Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-airflow.txt
```

### Step 5: Create Directory Structure

```bash
mkdir -p logs outputs config datasamples
chmod 755 logs outputs
chmod 700 config
```

---

## Configuration

### Environment Variables

Create `.env` file:

```bash
# Database
DB_HOST=your-db-server.database.windows.net
DB_PORT=5432
DB_NAME=card_analytics
DB_USER=anomaly_reader
DB_PASSWORD=your-secure-password

# Airflow
AIRFLOW_HOME=/opt/anomaly-detection/airflow
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://user:pass@localhost/airflow

# Alerting
SMTP_SERVER=smtp.bank.com
SMTP_PORT=587
SMTP_USER=anomaly-detection@bank.com
SMTP_PASSWORD=your-smtp-password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Application
LOG_LEVEL=INFO
OUTPUT_DIR=/opt/anomaly-detection/outputs
```

Set permissions:
```bash
chmod 600 .env
```

---

## Airflow Setup

### Step 1: Initialize Airflow

```bash
export AIRFLOW_HOME=/opt/anomaly-detection/airflow
airflow db init
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@bank.com --password your-secure-password
```

### Step 2: Configure Airflow

Edit `$AIRFLOW_HOME/airflow.cfg`:

```ini
[core]
dags_folder = /opt/anomaly-detection/dags
executor = LocalExecutor
load_examples = False

[database]
sql_alchemy_conn = postgresql+psycopg2://airflow:password@localhost/airflow

[smtp]
smtp_host = smtp.bank.com
smtp_starttls = True
smtp_port = 587
smtp_user = anomaly-detection@bank.com
smtp_password = your-smtp-password
smtp_mail_from = anomaly-detection@bank.com
```

### Step 3: Deploy DAG

```bash
cp dags/anomaly_detection_dag.py $AIRFLOW_HOME/dags/
airflow dags list
```

### Step 4: Start Airflow Services

Create systemd services for Airflow webserver and scheduler:

```bash
sudo systemctl enable airflow-webserver airflow-scheduler
sudo systemctl start airflow-webserver airflow-scheduler
```

---

## Monitoring Setup

### Configure Health Check Cron

```bash
crontab -e
# Add: 0 * * * * cd /opt/anomaly-detection && source venv/bin/activate && python -m src.monitoring.health_checks >> logs/health_checks.log 2>&1
```

### Test Monitoring

```bash
python -m src.monitoring.health_checks
python -m src.monitoring.alerting
```

---

## Power BI Integration

### Database Table Setup

```sql
CREATE TABLE anomaly_reports (
    customer_id VARCHAR(50) NOT NULL,
    reporting_week VARCHAR(7) NOT NULL,
    anomaly_score FLOAT NOT NULL,
    anomaly_rank INT NOT NULL,
    total_spend DECIMAL(12, 2),
    transaction_count INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id, reporting_week)
);
```

Configure Power BI Service to refresh daily at 8:00 AM CST.

---

## Deployment Checklist

### Pre-Deployment
- [ ] Server provisioned with required resources
- [ ] Dependencies installed
- [ ] Repository cloned
- [ ] Configuration files updated
- [ ] Database connectivity tested

### Deployment
- [ ] Airflow database initialized
- [ ] DAG deployed
- [ ] Airflow services started
- [ ] Monitoring configured
- [ ] Power BI data source configured

### Post-Deployment
- [ ] Trigger test DAG run
- [ ] Verify email alerts working
- [ ] Validate Power BI dashboard
- [ ] Notify stakeholders

---

## Runbook

### Daily Operations

**Morning Check (9:00 AM CST):**
1. Verify Monday's pipeline run completed
2. Review health check results
3. Check Power BI dashboard
4. Review alert emails

**Weekly Tasks (Tuesdays):**
1. Validate latest anomaly report
2. Review top anomalies with stakeholders
3. Archive old output files
4. Check disk space

### Common Tasks

**Manually Trigger Pipeline:**
```bash
cd /opt/anomaly-detection
source venv/bin/activate
python main.py --reporting-week $(date +%Y-%W) --output-dir outputs
```

**View Logs:**
```bash
tail -f logs/app.log
```

**Restart Airflow:**
```bash
sudo systemctl restart airflow-webserver airflow-scheduler
```

---

## Troubleshooting

### Issue: Airflow DAG Not Appearing

**Diagnosis:**
```bash
python $AIRFLOW_HOME/dags/anomaly_detection_dag.py
tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log
```

**Solution:**
1. Fix Python syntax errors
2. Ensure DAG file in correct directory
3. Restart scheduler

### Issue: Pipeline Execution Fails

**Common Causes:**
1. Database connection
2. Missing data
3. Disk space
4. Memory

**Solution:**
1. Verify credentials in `.env`
2. Check source data exists
3. Run `df -h` and `free -h`
4. Debug with `LOG_LEVEL=DEBUG python main.py`

### Issue: Email Alerts Not Sending

**Solution:**
1. Verify SMTP credentials
2. Check firewall allows port 587
3. Verify recipient addresses
4. Check spam folders

---

## Support

| Role | Contact | Responsibility |
|------|---------|----------------|
| DevOps Team | devops@bank.com | Infrastructure, Airflow |
| Data Science Team | data-science-team@bank.com | Model, data quality |
| BI Developer | bi-team@bank.com | Power BI dashboards |
| Database Admin | dba@bank.com | Database connectivity |

---

**Related Documentation:**
- [Requirements Specification](REQUIREMENTS.md)
- [Power BI Integration Guide](POWER_BI_GUIDE.md)
- [Implementation Tasks](tasks.md)

