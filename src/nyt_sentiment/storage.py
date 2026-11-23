"""Persistence helpers: PostgreSQL tables, S3-compatible object storage, MLflow.

All writes are idempotent so that a DAG can be re-run safely:
articles are upserted on ``article_id`` and each forecast run replaces the
previous one instead of appending duplicates.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    article_id  TEXT PRIMARY KEY,
    headline    TEXT,
    abstract    TEXT,
    section     TEXT,
    pub_date    TIMESTAMP NOT NULL,
    web_url     TEXT,
    label       TEXT,
    confidence  DOUBLE PRECISION,
    polarity    DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS sentiment_forecast (
    ds          DATE PRIMARY KEY,
    yhat        DOUBLE PRECISION NOT NULL,
    yhat_lower  DOUBLE PRECISION,
    yhat_upper  DOUBLE PRECISION,
    is_forecast BOOLEAN NOT NULL,
    run_at      TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

ARTICLE_TABLE_COLUMNS = [
    "article_id",
    "headline",
    "abstract",
    "section",
    "pub_date",
    "web_url",
    "label",
    "confidence",
    "polarity",
]


def get_engine(database_url: str):
    from sqlalchemy import create_engine

    return create_engine(database_url, pool_pre_ping=True)


def ensure_schema(engine) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        for statement in filter(None, (s.strip() for s in SCHEMA_SQL.split(";"))):
            conn.execute(text(statement))


def upsert_articles(engine, articles: pd.DataFrame) -> int:
    """Insert scored articles, updating existing rows on ``article_id``."""
    from sqlalchemy import MetaData, Table
    from sqlalchemy.dialects.postgresql import insert

    if articles.empty:
        return 0
    table = Table("articles", MetaData(), autoload_with=engine)
    rows = (
        articles[ARTICLE_TABLE_COLUMNS]
        .astype(object)
        .where(articles[ARTICLE_TABLE_COLUMNS].notna(), None)
        .to_dict("records")
    )
    statement = insert(table).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=["article_id"],
        set_={c: statement.excluded[c] for c in ARTICLE_TABLE_COLUMNS[1:]},
    )
    with engine.begin() as conn:
        conn.execute(statement)
    return len(rows)


def replace_forecast(engine, forecast: pd.DataFrame) -> int:
    """Replace the stored forecast with the latest run."""
    from sqlalchemy import text

    frame = forecast.assign(ds=pd.to_datetime(forecast["ds"]).dt.date)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM sentiment_forecast"))
        frame[["ds", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]].to_sql(
            "sentiment_forecast", conn, if_exists="append", index=False
        )
    return len(frame)


def existing_article_ids(engine) -> set[str]:
    from sqlalchemy import text

    with engine.connect() as conn:
        return {row[0] for row in conn.execute(text("SELECT article_id FROM articles"))}


def read_table(engine, name: str) -> pd.DataFrame:
    if name not in {"articles", "sentiment_forecast"}:
        raise ValueError(f"Unknown table {name!r}")
    return pd.read_sql_table(name, engine)


def s3_client(endpoint_url: str | None = None):
    import boto3

    return boto3.client("s3", endpoint_url=endpoint_url)


def download_json(client, bucket: str, key: str) -> list | dict:
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def ensure_bucket(client, bucket: str) -> None:
    existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    if bucket not in existing:
        client.create_bucket(Bucket=bucket)


def upload_json(client, bucket: str, key: str, payload: list | dict) -> str:
    """Store a raw API payload in object storage and return its URI."""
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


def log_forecast_run(
    tracking_uri: str,
    experiment: str,
    daily: pd.DataFrame,
    forecast: pd.DataFrame,
    params: dict,
) -> str:
    """Log parameters, summary metrics and the forecast table to MLflow."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    future = forecast.loc[forecast["is_forecast"]]
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "n_days": float(len(daily)),
                "n_articles": float(daily["n_articles"].sum()),
                "mean_sentiment_index": float(daily["sentiment_index"].mean()),
                "forecast_mean": float(future["yhat"].mean()),
                "forecast_interval_width": float(
                    (future["yhat_upper"] - future["yhat_lower"]).mean()
                ),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forecast.csv"
            forecast.to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="forecast")
        return run.info.run_id
