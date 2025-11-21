"""
Airflow DAG for Card Portfolio Anomaly Detection System.

This DAG orchestrates the weekly anomaly detection pipeline:
- Runs every Monday at 2:00 AM CST
- Sends email alerts on failure
- Calls main pipeline with reporting week

Authority: Platform (REQ-7.1)
Owner: DevOps Team
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago


# Default arguments for the DAG
DEFAULT_ARGS: Dict[str, Any] = {
    "owner": "data-science-team",
    "depends_on_past": False,
    "email": ["data-science-team@bank.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}


def run_anomaly_detection(**context) -> None:
    """Execute the anomaly detection pipeline for the reporting week."""
    import subprocess
    import sys
    from pathlib import Path

    execution_date = context["execution_date"]
    reporting_week = execution_date.strftime("%Y-%W")
    
    project_root = Path(__file__).parent.parent
    main_script = project_root / "main.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(main_script), "--reporting-week", reporting_week],
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        print(f"Pipeline completed for week {reporting_week}")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Pipeline failed: {e.stderr}") from e


# Define the DAG
with DAG(
    dag_id="card_portfolio_anomaly_detection",
    default_args=DEFAULT_ARGS,
    description="Weekly anomaly detection for card portfolio customers",
    schedule_interval="0 7 * * 1",  # Every Monday at 7:00 AM UTC (2:00 AM CST)
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["anomaly-detection", "card-portfolio", "weekly"],
) as dag:
    
    detect_anomalies = PythonOperator(
        task_id="run_anomaly_detection_pipeline",
        python_callable=run_anomaly_detection,
        provide_context=True,
    )
    
    notify_success = EmailOperator(
        task_id="send_success_notification",
        to=["data-science-team@bank.com"],
        subject="✅ Anomaly Detection Pipeline Completed - {{ ds }}",
        html_content="<h3>Pipeline completed successfully</h3>",
        trigger_rule="all_success",
    )
    
    detect_anomalies >> notify_success

