"""
Airflow DAG for Card Portfolio Anomaly Detection System.

This DAG orchestrates the weekly execution of the anomaly detection pipeline.
Runs every Monday at 2:00 AM CST (8:00 AM UTC).
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.email import EmailOperator
from airflow.operators.python import PythonOperator


def run_anomaly_detection(**context) -> None:
    """Execute the anomaly detection pipeline."""
    import subprocess

    reporting_week = context["ds"]  # Execution date
    cmd = [
        "python",
        "/opt/anomaly-detection/main.py",
        "--reporting-week",
        reporting_week,
    ]
    subprocess.run(cmd, check=True)


default_args = {
    "owner": "data-science-team",
    "depends_on_past": False,
    "email": ["data-science-team@bank.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}

with DAG(
    "card_portfolio_anomaly_detection",
    default_args=default_args,
    description="Weekly anomaly detection for card portfolios",
    schedule_interval="0 8 * * 1",  # Monday 8AM UTC = 2AM CST
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["anomaly-detection", "card-portfolio"],
) as dag:

    run_pipeline = PythonOperator(
        task_id="run_anomaly_detection",
        python_callable=run_anomaly_detection,
    )

    notify_success = EmailOperator(
        task_id="notify_success",
        to="data-science-team@bank.com",
        subject="Anomaly Detection Pipeline Completed",
        html_content=(
            "<p>The weekly anomaly detection pipeline has completed successfully.</p>"
        ),
    )

    run_pipeline >> notify_success
