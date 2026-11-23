"""End-to-end steps shared by the Airflow DAG and the command line.

Database mode (Airflow, local stack):
    fetch_raw_to_s3 -> score_new_articles -> forecast_and_store

Each step reads its inputs from durable storage (object storage or
PostgreSQL) rather than from local temporary files, so tasks can run on
different workers and be retried independently.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from nyt_sentiment import forecast, nyt_api, preprocess, sentiment, storage
from nyt_sentiment.config import Settings

log = logging.getLogger(__name__)


def fetch_raw_to_s3(settings: Settings, months: list[tuple[int, int]]) -> list[str]:
    """Download Archive API months and store the raw payloads in S3."""
    client = storage.s3_client(settings.s3_endpoint_url)
    storage.ensure_bucket(client, settings.s3_bucket)
    keys = []
    for index, (year, month) in enumerate(months):
        if index:
            time.sleep(12)  # NYT allows 5 requests per minute
        docs = nyt_api.fetch_archive_month(year, month, settings.nyt_api_key)
        key = f"raw/archive/{year}-{month:02d}.json"
        storage.upload_json(client, settings.s3_bucket, key, docs)
        log.info("Stored %d raw documents at s3://%s/%s", len(docs), settings.s3_bucket, key)
        keys.append(key)
    return keys


def load_articles_from_s3(settings: Settings, keys: list[str]) -> pd.DataFrame:
    client = storage.s3_client(settings.s3_endpoint_url)
    docs = []
    for key in keys:
        docs.extend(storage.download_json(client, settings.s3_bucket, key))
    return preprocess.clean_articles(nyt_api.docs_to_frame(docs))


def score_new_articles(
    settings: Settings, keys: list[str], classifier: sentiment.Classifier | None = None
) -> int:
    """Score articles not yet in the database and upsert them."""
    engine = storage.get_engine(settings.database_url)
    storage.ensure_schema(engine)
    articles = load_articles_from_s3(settings, keys)
    new = articles.loc[~articles["article_id"].isin(storage.existing_article_ids(engine))]
    log.info("%d articles cleaned, %d new to score", len(articles), len(new))
    if new.empty:
        return 0
    classifier = classifier or sentiment.load_classifier(settings.sentiment_model)
    return storage.upsert_articles(engine, sentiment.score_articles(new, classifier))


def forecast_and_store(settings: Settings) -> dict:
    """Rebuild the daily index, forecast it, store it and log the run."""
    engine = storage.get_engine(settings.database_url)
    articles = storage.read_table(engine, "articles")
    daily = forecast.daily_sentiment(articles)
    predicted = forecast.forecast_index(daily, settings.forecast_horizon_days)
    storage.replace_forecast(engine, predicted)

    summary = {"n_days": len(daily), "n_articles": int(daily["n_articles"].sum())}
    if settings.mlflow_tracking_uri:
        summary["mlflow_run_id"] = storage.log_forecast_run(
            settings.mlflow_tracking_uri,
            settings.mlflow_experiment,
            daily,
            predicted,
            params={
                "model": settings.sentiment_model,
                "horizon_days": settings.forecast_horizon_days,
                "first_day": str(daily["date"].min().date()),
                "last_day": str(daily["date"].max().date()),
            },
        )
    return summary
