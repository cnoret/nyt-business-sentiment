"""Load the dashboard data from one of three sources, in this order:

1. ``DATABASE_URL``: PostgreSQL tables written by the Airflow pipeline
   (local docker compose stack);
2. ``DATA_URL``: base URL of a published Parquet snapshot (the public app
   reads the ``data`` branch of the repository);
3. ``DATA_DIR``: a local folder holding the same snapshot (default ``data``).
"""

from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyt_sentiment.forecast import daily_sentiment  # noqa: E402


@dataclass
class Dashboard:
    articles: pd.DataFrame
    daily: pd.DataFrame
    forecast: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    source: str = ""


def _from_database(url: str) -> Dashboard:
    from sqlalchemy import create_engine

    engine = create_engine(url)
    articles = pd.read_sql_table("articles", engine)
    forecast = pd.read_sql_table("sentiment_forecast", engine)
    engine.dispose()
    daily = daily_sentiment(articles)
    metadata = {
        "updated_at": str(forecast["run_at"].max()) if "run_at" in forecast else None,
        "n_articles": len(articles),
        "first_day": str(daily["date"].min().date()),
        "last_day": str(daily["date"].max().date()),
    }
    return Dashboard(articles, daily, forecast, metadata, "PostgreSQL")


def _from_url(base: str) -> Dashboard:
    import requests

    def get(name: str) -> bytes:
        response = requests.get(f"{base.rstrip('/')}/{name}", timeout=30)
        response.raise_for_status()
        return response.content

    return Dashboard(
        pd.read_parquet(io.BytesIO(get("articles.parquet"))),
        pd.read_parquet(io.BytesIO(get("daily.parquet"))),
        pd.read_parquet(io.BytesIO(get("forecast.parquet"))),
        json.loads(get("metadata.json")),
        "published snapshot",
    )


def _from_dir(path: Path) -> Dashboard:
    return Dashboard(
        pd.read_parquet(path / "articles.parquet"),
        pd.read_parquet(path / "daily.parquet"),
        pd.read_parquet(path / "forecast.parquet"),
        json.loads((path / "metadata.json").read_text()),
        "local snapshot",
    )


def load_dashboard() -> Dashboard:
    if url := os.getenv("DATABASE_URL"):
        board = _from_database(url)
    elif base := os.getenv("DATA_URL"):
        board = _from_url(base)
    else:
        board = _from_dir(Path(os.getenv("DATA_DIR", "data")))
    for frame, column in (
        (board.articles, "pub_date"),
        (board.daily, "date"),
        (board.forecast, "ds"),
    ):
        frame[column] = pd.to_datetime(frame[column])
    return board
