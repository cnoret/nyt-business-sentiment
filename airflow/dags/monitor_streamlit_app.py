"""Check every 30 minutes that the Streamlit dashboard answers.

An optional Slack incoming webhook (``SLACK_WEBHOOK_URL``) is notified when
the check fails.
"""

from __future__ import annotations

import os

import pendulum
import requests
from airflow.sdk import dag, task


def _alert_slack(context) -> None:
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if webhook:
        url = os.getenv("STREAMLIT_APP_URL", "")
        requests.post(webhook, json={"text": f":warning: Streamlit app is down: {url}"}, timeout=10)


@dag(
    dag_id="monitor_streamlit_app",
    schedule="*/30 * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["monitoring"],
)
def monitor_streamlit_app():
    @task(retries=1, on_failure_callback=_alert_slack)
    def check_streamlit_app() -> int:
        url = os.environ["STREAMLIT_APP_URL"]
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.status_code

    check_streamlit_app()


monitor_streamlit_app()
