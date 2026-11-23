"""File-based mode: keep a Parquet snapshot of scored articles and forecasts.

Used by the scheduled GitHub Actions job that feeds the public Streamlit app,
so the live demo needs no database, object storage or server.

Files written to ``data_dir``:
- ``articles.parquet``: every scored business article (incrementally grown)
- ``daily.parquet``: daily sentiment index, shares and article counts
- ``forecast.parquet``: Prophet fit on history plus the forecast horizon
- ``metadata.json``: refresh time, coverage and model used
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nyt_sentiment import forecast, nyt_api, preprocess, sentiment
from nyt_sentiment.config import Settings

log = logging.getLogger(__name__)

ARTICLES_FILE = "articles.parquet"
DAILY_FILE = "daily.parquet"
FORECAST_FILE = "forecast.parquet"
METADATA_FILE = "metadata.json"


def load_articles(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / ARTICLES_FILE
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def merge_articles(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append newly scored articles, keeping the latest version of each id."""
    merged = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    merged = merged.drop_duplicates(subset="article_id", keep="last")
    return merged.sort_values("pub_date").reset_index(drop=True)


def update_snapshot(
    settings: Settings,
    months: list[tuple[int, int]],
    classifier: sentiment.Classifier | None = None,
    fetch=nyt_api.fetch_archive_month,
    now: datetime | None = None,
) -> dict:
    """Fetch ``months``, score unseen articles and rewrite every snapshot file."""
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = load_articles(data_dir)
    seen = set(existing["article_id"]) if not existing.empty else set()

    docs = []
    for year, month in months:
        month_docs = fetch(year, month, settings.nyt_api_key)
        log.info("Fetched %d documents for %d-%02d", len(month_docs), year, month)
        docs.extend(month_docs)

    cleaned = preprocess.clean_articles(nyt_api.docs_to_frame(docs))
    new = cleaned.loc[~cleaned["article_id"].isin(seen)]
    log.info("%d business articles, %d not scored yet", len(cleaned), len(new))

    if not new.empty:
        classifier = classifier or sentiment.load_classifier(settings.sentiment_model)
        new = sentiment.score_articles(new, classifier).drop(columns="text")
    articles = merge_articles(existing, new)
    if articles.empty:
        raise RuntimeError("No articles available: nothing to aggregate.")

    daily = forecast.daily_sentiment(articles)
    predicted = forecast.forecast_index(daily, settings.forecast_horizon_days)

    articles.to_parquet(data_dir / ARTICLES_FILE, index=False)
    daily.to_parquet(data_dir / DAILY_FILE, index=False)
    predicted.to_parquet(data_dir / FORECAST_FILE, index=False)

    metadata = {
        "updated_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "n_articles": int(len(articles)),
        "n_new_articles": int(len(new)),
        "first_day": str(daily["date"].min().date()),
        "last_day": str(daily["date"].max().date()),
        "horizon_days": settings.forecast_horizon_days,
        "model": settings.sentiment_model,
    }
    (data_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
