"""Integration tests against a real PostgreSQL (skipped without DATABASE_URL)."""

import os

import pandas as pd
import pytest

from nyt_sentiment import storage

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
def engine():
    from sqlalchemy import text

    eng = storage.get_engine(DATABASE_URL)
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS articles, sentiment_forecast"))
    storage.ensure_schema(eng)
    yield eng
    eng.dispose()


def _article(article_id, label="Neutral", polarity=0.0):
    return {
        "article_id": article_id,
        "headline": "Headline",
        "abstract": None,
        "section": "Business",
        "pub_date": pd.Timestamp("2026-01-05 14:00"),
        "web_url": "https://www.nytimes.com/x",
        "label": label,
        "confidence": 0.9,
        "polarity": polarity,
    }


def test_upsert_articles_is_idempotent(engine):
    storage.upsert_articles(engine, pd.DataFrame([_article("a"), _article("b")]))
    storage.upsert_articles(engine, pd.DataFrame([_article("a", "Positive", 0.5)]))
    stored = storage.read_table(engine, "articles").set_index("article_id")
    assert len(stored) == 2
    assert stored.loc["a", "label"] == "Positive"
    assert pd.isna(stored.loc["b", "abstract"])


def test_replace_forecast_does_not_duplicate(engine):
    forecast = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=3),
            "yhat": [0.1, 0.2, 0.3],
            "yhat_lower": [0.0, 0.1, 0.2],
            "yhat_upper": [0.2, 0.3, 0.4],
            "is_forecast": [False, False, True],
        }
    )
    storage.replace_forecast(engine, forecast)
    storage.replace_forecast(engine, forecast)
    assert len(storage.read_table(engine, "sentiment_forecast")) == 3
