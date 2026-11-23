"""Weekly NYT business sentiment pipeline.

fetch_raw_articles -> score_new_articles -> forecast_sentiment

Raw API payloads land in S3-compatible storage (MinIO locally), scored
articles and the forecast in PostgreSQL, and every forecast run is tracked in
MLflow. All logic lives in the ``nyt_sentiment`` package; this file only wires
the steps together.
"""

from __future__ import annotations

import pendulum
from airflow.sdk import dag, task

from nyt_sentiment.config import get_settings
from nyt_sentiment.nyt_api import months_between


@dag(
    dag_id="nyt_sentiment_pipeline",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=5)},
    params={"lookback_months": 1},
    tags=["nyt", "sentiment", "forecast"],
)
def nyt_sentiment_pipeline():
    @task
    def fetch_raw_articles(data_interval_end=None, params=None) -> list[str]:
        from nyt_sentiment.pipeline import fetch_raw_to_s3

        end = pendulum.instance(data_interval_end)
        start = end.subtract(months=int(params["lookback_months"]))
        return fetch_raw_to_s3(get_settings(), months_between(start, end))

    @task
    def score_new_articles(keys: list[str]) -> int:
        from nyt_sentiment.pipeline import score_new_articles as score

        return score(get_settings(), keys)

    @task
    def forecast_sentiment(_n_scored: int) -> dict:
        from nyt_sentiment.pipeline import forecast_and_store

        return forecast_and_store(get_settings())

    forecast_sentiment(score_new_articles(fetch_raw_articles()))


nyt_sentiment_pipeline()
