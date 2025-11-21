# Deployment Guide: Card Portfolio Anomaly Detection System

**Version:** 1.0.0  
**Status:** Production  
**Owner:** DevOps Team + Lead Developer  
**Date:** 2025-11-21  
**Authority Level:** Platform (REQ-7.4)

---

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
10. [Rollback Procedures](#rollback-procedures)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This guide provides step-by-step instructions for deploying the Card Portfolio Anomaly Detection System to production.

### Deployment Architecture

```
┌─────────────────┐
│   Data Source   │
│   (Database)    │
└────────┬────────┘
         │
         v
┌─────────────────┐      ┌──────────────┐
│  Airflow        │─────>│  Pipeline    │
│  Scheduler      │      │  Execution   │
└─────────────────┘      └──────┬───────┘
                                │
                                v
                         ┌──────────────┐
                         │  Outputs     │
                         │  (CSV/JSON)  │
                         └──────┬───────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                v               v               v
         ┌──────────┐    ┌──────────┐   ┌──────────┐
         │ Power BI │    │Monitoring│   │  Email   │
         │Dashboard │    │ Alerts   │   │  Reports │
         └──────────┘    └──────────┘   └──────────┘
```

### Environments

- **Development:** Local development and testing
- **Staging:** Pre-production validation
- **Production:** Live system serving stakeholders

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| Disk | 100 GB | 250 GB |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 LTS |
| Python | 3.9+ | 3.11+ |

### Software Dependencies

- Python 3.9 or higher
- Git 2.30+
- PostgreSQL 13+ (or compatible database)
- Apache Airflow 2.8+
- SMTP server access (for email alerts)
- (Optional) Docker 20.10+ for containerized deployment

### Access Requirements

- Database credentials (read-only for source data)
- SMTP credentials for email alerts
- Slack webhook URL (if using Slack alerts)
- Power BI service credentials
- SSH access to deployment server

### Network Requirements

- Outbound HTTPS (port 443) for email/Slack
- Database port access (typically 5432 for PostgreSQL)
- (Optional) Grafana/Prometheus port access

---

## Server Setup

### Step 1: Create Deployment User

```bash
# Create a dedicated user for the anomaly detection system
sudo useradd -m -s /bin/bash anomaly-detection
sudo usermod -aG sudo anomaly-detection

# Switch to the deployment user
sudo su - anomaly-detection
```

### Step 2: Install System Dependencies

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install Python and build tools
sudo apt install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    libpq-dev

# Install optional monitoring tools
sudo apt install -y htop iotop ncdu
```

### Step 3: Clone Repository

```bash
# Create application directory
mkdir -p /opt/anomaly-detection
cd /opt/anomaly-detection

# Clone the repository
git clone https://github.com/your-org/card-portfolio-anomaly-detection.git .

# Checkout the desired version/tag
git checkout v1.0.0
```

### Step 4: Create Python Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install project dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-airflow.txt
```

### Step 5: Create Directory Structure

```bash
# Create required directories
mkdir -p logs outputs config datasamples

# Set proper permissions
chmod 755 logs outputs
chmod 700 config  # Sensitive config files
```

---

## Configuration

### Step 1: Environment Variables

Create `.env` file (copy from `.env.example`):

```bash
# Copy example file
cp .env.example .env

# Edit with production values
nano .env
```

**Required Environment Variables:**

```bash
# Database Configuration
DB_HOST=your-db-server.database.windows.net
DB_PORT=5432
DB_NAME=card_analytics
DB_USER=anomaly_reader
DB_PASSWORD=your-secure-password

# Airflow Configuration
AIRFLOW_HOME=/opt/anomaly-detection/airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://user:pass@localhost/airflow
AIRFLOW__CORE__LOAD_EXAMPLES=False

# Alerting Configuration
SMTP_SERVER=smtp.bank.com
SMTP_PORT=587
SMTP_USER=anomaly-detection@bank.com
SMTP_PASSWORD=your-smtp-password

SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Monitoring Configuration
ENABLE_MONITORING=true
ALERT_THRESHOLD=warn

# Application Configuration
LOG_LEVEL=INFO
OUTPUT_DIR=/opt/anomaly-detection/outputs
```

**Security Note:** Ensure `.env` file permissions are restricted:

```bash
chmod 600 .env
```

### Step 2: Application Configuration

Edit `config/config.yaml` with production settings:

```yaml
# Production configuration
environment: production
data_source:
  type: database
  connection_string: "${DB_CONNECTION_STRING}"
feature_engineering:
  use_cache: true
  cache_dir: /opt/anomaly-detection/cache
model:
  contamination: 0.05
  random_state: 42
output:
  directory: /opt/anomaly-detection/outputs
  format: csv
  retention_days: 90
logging:
  level: INFO
  file: /opt/anomaly-detection/logs/app.log
  max_size_mb: 100
  backup_count: 10
```

### Step 3: Validate Configuration

```bash
# Test configuration loading
python -c "from src.config import load_config; print(load_config())"

# Test database connection
python -c "from src.data.loader import DataLoader; loader = DataLoader(); loader.test_connection()"
```

---

## Airflow Setup

### Step 1: Initialize Airflow

```bash
# Set Airflow home
export AIRFLOW_HOME=/opt/anomaly-detection/airflow

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@bank.com \
    --password your-secure-password
```

### Step 2: Configure Airflow

Edit `$AIRFLOW_HOME/airflow.cfg`:

```ini
[core]
dags_folder = /opt/anomaly-detection/dags
base_log_folder = /opt/anomaly-detection/logs/airflow
executor = LocalExecutor
load_examples = False

[database]
sql_alchemy_conn = postgresql+psycopg2://airflow:password@localhost/airflow

[smtp]
smtp_host = smtp.bank.com
smtp_starttls = True
smtp_ssl = False
smtp_user = anomaly-detection@bank.com
smtp_password = your-smtp-password
smtp_port = 587
smtp_mail_from = anomaly-detection@bank.com

[scheduler]
catchup_by_default = False
max_active_runs_per_dag = 1
```

### Step 3: Deploy DAG

```bash
# Copy DAG to Airflow DAGs folder
cp dags/anomaly_detection_dag.py $AIRFLOW_HOME/dags/

# Validate DAG
airflow dags list
airflow dags list-runs -d card_portfolio_anomaly_detection
```

### Step 4: Start Airflow Services

**Option A: Systemd Services (Recommended)**

Create `/etc/systemd/system/airflow-webserver.service`:

```ini
[Unit]
Description=Airflow Webserver
After=network.target

[Service]
Type=simple
User=anomaly-detection
Group=anomaly-detection
Environment="AIRFLOW_HOME=/opt/anomaly-detection/airflow"
Environment="PATH=/opt/anomaly-detection/venv/bin"
ExecStart=/opt/anomaly-detection/venv/bin/airflow webserver --port 8080
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/airflow-scheduler.service`:

```ini
[Unit]
Description=Airflow Scheduler
After=network.target

[Service]
Type=simple
User=anomaly-detection
Group=anomaly-detection
Environment="AIRFLOW_HOME=/opt/anomaly-detection/airflow"
Environment="PATH=/opt/anomaly-detection/venv/bin"
ExecStart=/opt/anomaly-detection/venv/bin/airflow scheduler
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start services:

```bash
sudo systemctl daemon-reload
sudo systemctl enable airflow-webserver airflow-scheduler
sudo systemctl start airflow-webserver airflow-scheduler

# Check status
sudo systemctl status airflow-webserver
sudo systemctl status airflow-scheduler
```

**Option B: Manual Start (Development)**

```bash
# Start webserver
airflow webserver --port 8080 &

# Start scheduler
airflow scheduler &
```

### Step 5: Verify Airflow

```bash
# Access web UI
# Navigate to: http://your-server:8080
# Login with admin credentials

# Trigger test run
airflow dags trigger card_portfolio_anomaly_detection

# View logs
airflow tasks logs card_portfolio_anomaly_detection run_anomaly_detection_pipeline <execution_date>
```

---

## Monitoring Setup

### Step 1: Configure Monitoring

Edit `config/monitoring_config.yaml` with production settings (see config file for details).

### Step 2: Set Up Health Check Cron

```bash
# Add cron job for health checks
crontab -e

# Add this line to run health checks every hour
0 * * * * cd /opt/anomaly-detection && source venv/bin/activate && python -m src.monitoring.health_checks >> logs/health_checks.log 2>&1
```

### Step 3: Test Monitoring

```bash
# Run health checks manually
python -m src.monitoring.health_checks

# Send test alert
python -m src.monitoring.alerting
```

### Step 4: (Optional) Set Up Grafana Dashboard

If using Grafana/Prometheus:

1. Install Prometheus Node Exporter
2. Configure scrape targets in Prometheus
3. Import Grafana dashboard template
4. Configure alert rules

---

## Power BI Integration

### Step 1: Database Table Setup

Create a table for Power BI to query:

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
```

### Step 2: Configure Data Pipeline to Load Database

Modify the pipeline to insert results into the database table.

### Step 3: Configure Power BI Data Source

Follow the instructions in `docs/POWER_BI_GUIDE.md`.

### Step 4: Schedule Dashboard Refresh

Configure Power BI Service to refresh daily at 8:00 AM CST.

---

## Deployment Checklist

### Pre-Deployment

- [ ] Server provisioned with required resources
- [ ] All dependencies installed
- [ ] Repository cloned and checked out to correct version
- [ ] Virtual environment created and dependencies installed
- [ ] Configuration files updated with production values
- [ ] Environment variables set in `.env` file
- [ ] Database connectivity tested
- [ ] SMTP credentials validated
- [ ] Slack webhook URL configured (if applicable)

### Deployment

- [ ] Airflow database initialized
- [ ] Airflow admin user created
- [ ] DAG deployed to Airflow
- [ ] Airflow services started (webserver + scheduler)
- [ ] Health check cron job configured
- [ ] Monitoring alerts tested
- [ ] Power BI data source configured
- [ ] Power BI dashboard published

### Post-Deployment

- [ ] Trigger test DAG run and verify success
- [ ] Check output files generated correctly
- [ ] Verify email alerts working
- [ ] Verify Slack alerts working (if configured)
- [ ] Validate Power BI dashboard displays data
- [ ] Document deployment details (date, version, deployer)
- [ ] Update runbook with any deviations
- [ ] Notify stakeholders of deployment completion

---

## Runbook

### Daily Operations

**Morning Check (9:00 AM CST):**

1. Verify Monday's pipeline run completed successfully (check Airflow UI)
2. Review health check results in `outputs/health_check_latest.json`
3. Check Power BI dashboard for latest data
4. Review any alert emails from overnight

**Weekly Tasks (Tuesdays):**

1. Validate latest anomaly report in `outputs/anomalies_YYYY-WW.csv`
2. Review top 20 anomalies with business stakeholders
3. Archive old output files (keep last 13 weeks)
4. Check disk space usage

### Common Tasks

**Manually Trigger Pipeline:**

```bash
cd /opt/anomaly-detection
source venv/bin/activate
python main.py --reporting-week $(date +%Y-%W) --output-dir outputs
```

**View Recent Logs:**

```bash
tail -f logs/app.log
tail -f logs/airflow/scheduler/latest/card_portfolio_anomaly_detection/*.log
```

**Check Airflow Status:**

```bash
sudo systemctl status airflow-webserver
sudo systemctl status airflow-scheduler
```

**Restart Airflow:**

```bash
sudo systemctl restart airflow-webserver
sudo systemctl restart airflow-scheduler
```

**Run Health Checks:**

```bash
cd /opt/anomaly-detection
source venv/bin/activate
python -m src.monitoring.health_checks
```

---

## Rollback Procedures

### Scenario 1: Bad Deployment (Code Issues)

**Steps:**

1. Stop Airflow services:
   ```bash
   sudo systemctl stop airflow-webserver airflow-scheduler
   ```

2. Checkout previous stable version:
   ```bash
   cd /opt/anomaly-detection
   git fetch --tags
   git checkout v0.9.0  # Previous stable version
   ```

3. Reinstall dependencies (if changed):
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. Restart Airflow services:
   ```bash
   sudo systemctl start airflow-webserver airflow-scheduler
   ```

5. Verify rollback:
   ```bash
   airflow dags trigger card_portfolio_anomaly_detection
   # Monitor execution
   ```

### Scenario 2: Configuration Issues

**Steps:**

1. Restore previous configuration:
   ```bash
   cp config/config.yaml.backup config/config.yaml
   cp .env.backup .env
   ```

2. Restart services:
   ```bash
   sudo systemctl restart airflow-scheduler
   ```

3. Validate with test run

### Scenario 3: Database Issues

**Steps:**

1. Switch to backup data source (if available)
2. Update connection string in `.env`
3. Restart Airflow scheduler
4. Notify database team to investigate primary source

---

## Troubleshooting

### Issue: Airflow DAG Not Appearing

**Symptoms:** DAG not visible in Airflow UI

**Diagnosis:**

```bash
# Check DAG for syntax errors
python $AIRFLOW_HOME/dags/anomaly_detection_dag.py

# Check Airflow logs
tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log
```

**Solution:**

1. Fix any Python syntax errors
2. Ensure DAG file is in `$AIRFLOW_HOME/dags/` directory
3. Restart scheduler: `sudo systemctl restart airflow-scheduler`

---

### Issue: Pipeline Execution Fails

**Symptoms:** DAG run marked as failed in Airflow

**Diagnosis:**

```bash
# Check task logs in Airflow UI
# Or view logs directly:
tail -f logs/app.log
```

**Common Causes:**

1. **Database Connection:** Verify credentials in `.env`
2. **Missing Data:** Check source data exists for reporting week
3. **Disk Space:** Run `df -h` to check available space
4. **Memory:** Run `free -h` to check available RAM

**Solution:**

1. Address root cause based on error message
2. Retry failed task in Airflow UI
3. If persistent, manually run pipeline with debug logging:
   ```bash
   LOG_LEVEL=DEBUG python main.py --reporting-week YYYY-WW
   ```

---

### Issue: Email Alerts Not Sending

**Symptoms:** No email alerts received on pipeline failure

**Diagnosis:**

```bash
# Test SMTP connection
python -c "
import smtplib
server = smtplib.SMTP('smtp.bank.com', 587)
server.starttls()
server.login('user', 'password')
print('SMTP connection successful')
server.quit()
"
```

**Solution:**

1. Verify SMTP credentials in `.env`
2. Check firewall allows outbound port 587
3. Verify recipient email addresses are correct
4. Check spam/junk folders

---

### Issue: Power BI Dashboard Not Refreshing

**Symptoms:** Dashboard shows stale data

**Diagnosis:**

1. Check Power BI Service refresh history
2. Verify output files exist: `ls -lh outputs/anomalies_*.csv`
3. Check database table has latest data (if using DB source)

**Solution:**

1. Manually trigger Power BI refresh
2. Verify data source credentials in Power BI Service
3. Check gateway status (if using on-premises data)
4. Review refresh logs for specific errors

---

### Issue: High CPU/Memory Usage

**Symptoms:** Server performance degraded, slow pipeline execution

**Diagnosis:**

```bash
# Check resource usage
htop
iotop
df -h
```

**Solution:**

1. **High CPU:** 
   - Check if multiple DAG runs executing concurrently
   - Set `max_active_runs_per_dag = 1` in Airflow config
   
2. **High Memory:**
   - Reduce batch size in data processing
   - Optimize pandas operations (use chunking)
   - Increase server RAM if persistent

3. **Disk Space:**
   - Clean old log files: `find logs/ -name "*.log" -mtime +30 -delete`
   - Archive old outputs: `tar -czf outputs_archive.tar.gz outputs/anomalies_*.csv`
   - Increase disk allocation

---

## Support and Contacts

| Role | Contact | Responsibility |
|------|---------|----------------|
| DevOps Team | devops@bank.com | Infrastructure, Airflow, monitoring |
| Data Science Team | data-science-team@bank.com | Model, features, data quality |
| BI Developer | bi-team@bank.com | Power BI dashboards, reporting |
| Database Admin | dba@bank.com | Database connectivity, performance |
| Security Team | security@bank.com | Access, credentials, compliance |

---

## Appendix A: Deployment Automation Script

```bash
#!/bin/bash
# deploy.sh - Automated deployment script

set -e  # Exit on error

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: ./deploy.sh <version>"
    exit 1
fi

echo "=== Deploying version $VERSION ==="

# Step 1: Backup current state
echo "Backing up current deployment..."
cp config/config.yaml config/config.yaml.backup
cp .env .env.backup

# Step 2: Checkout new version
echo "Checking out version $VERSION..."
git fetch --tags
git checkout "$VERSION"

# Step 3: Update dependencies
echo "Updating dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# Step 4: Run tests
echo "Running tests..."
pytest tests/ -v

# Step 5: Update Airflow DAG
echo "Updating Airflow DAG..."
cp dags/anomaly_detection_dag.py $AIRFLOW_HOME/dags/

# Step 6: Restart services
echo "Restarting services..."
sudo systemctl restart airflow-scheduler

# Step 7: Verify deployment
echo "Verifying deployment..."
sleep 10
airflow dags list | grep card_portfolio_anomaly_detection

echo "=== Deployment complete ==="
echo "Please verify:"
echo "1. Airflow UI shows DAG"
echo "2. Trigger test run"
echo "3. Monitor logs for errors"
```

---

## Appendix B: Monitoring Checklist

**Daily:**
- [ ] Check Airflow UI for successful DAG runs
- [ ] Review health check results
- [ ] Verify Power BI dashboard updated

**Weekly:**
- [ ] Review anomaly trends
- [ ] Archive old output files
- [ ] Check disk space usage

**Monthly:**
- [ ] Review system performance metrics
- [ ] Update documentation if needed
- [ ] Audit user access and permissions
- [ ] Review and optimize configurations

**Quarterly:**
- [ ] Review and update runbook
- [ ] Conduct disaster recovery drill
- [ ] Evaluate system resource requirements
- [ ] Plan capacity upgrades if needed

---

**Document Change Log**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-21 | Codegen Bot | Initial deployment guide |

---

**Related Documentation:**

- [Requirements Specification](REQUIREMENTS.md)
- [System Design](design.md)
- [Power BI Integration Guide](POWER_BI_GUIDE.md)
- [Implementation Tasks](tasks.md)

