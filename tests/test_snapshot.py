import json
from datetime import UTC, datetime

import pandas as pd

from nyt_sentiment.cli import recent_months
from nyt_sentiment.config import Settings
from nyt_sentiment.snapshot import load_articles, update_snapshot


def _month_docs(year, month):
    days = pd.date_range(f"{year}-{month:02d}-01", periods=20, freq="D")
    docs = []
    for i, day in enumerate(days):
        for j, word in enumerate(("rally", "closes", "steady")):
            docs.append(
                {
                    "uri": f"nyt://{year}-{month}-{i}-{j}",
                    "headline": {"main": f"Markets {word} on day {i}"},
                    "abstract": "Business news.",
                    "section_name": "Business",
                    "pub_date": (day + pd.Timedelta(hours=j)).isoformat() + "Z",
                    "web_url": "https://www.nytimes.com/x",
                }
            )
    return docs


def test_update_snapshot_is_incremental(tmp_path, fake_classifier):
    settings = Settings(data_dir=tmp_path, forecast_horizon_days=7, nyt_api_key="k")

    def fetch(year, month, _key):
        return _month_docs(year, month)

    now = datetime(2026, 2, 21, tzinfo=UTC)
    meta = update_snapshot(settings, [(2026, 1)], fake_classifier, fetch, now=now)
    assert meta["n_articles"] == 60
    assert meta["n_new_articles"] == 60

    meta = update_snapshot(settings, [(2026, 1), (2026, 2)], fake_classifier, fetch, now=now)
    assert meta["n_articles"] == 120
    assert meta["n_new_articles"] == 60  # January was not scored twice
    assert meta["last_day"] == "2026-02-20"

    articles = load_articles(tmp_path)
    assert articles["article_id"].is_unique
    assert "text" not in articles.columns
    forecast = pd.read_parquet(tmp_path / "forecast.parquet")
    assert forecast["is_forecast"].sum() == 7
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["updated_at"] == "2026-02-21T00:00:00+00:00"


def test_recent_months_includes_current_month():
    today = pd.Timestamp("2026-03-15", tz="UTC")
    assert recent_months(3, today) == [(2026, 1), (2026, 2), (2026, 3)]
