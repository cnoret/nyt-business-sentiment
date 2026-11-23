"""Smoke test of the dashboard on a synthetic snapshot (Streamlit AppTest)."""

from pathlib import Path

import pytest

from nyt_sentiment.config import Settings
from nyt_sentiment.snapshot import update_snapshot

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.test_snapshot import _month_docs  # noqa: E402


@pytest.fixture(autouse=True)
def clear_cache():
    import streamlit as st

    st.cache_data.clear()


APP = Path(__file__).resolve().parents[1] / "streamlit" / "app.py"


@pytest.fixture
def snapshot_dir(tmp_path, fake_classifier):
    settings = Settings(data_dir=tmp_path, forecast_horizon_days=30, nyt_api_key="k")
    update_snapshot(
        settings,
        [(2026, 1), (2026, 2)],
        fake_classifier,
        fetch=lambda year, month, _key: _month_docs(year, month),
    )
    return tmp_path


def test_dashboard_renders(monkeypatch, snapshot_dir):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATA_URL", raising=False)
    monkeypatch.setenv("DATA_DIR", str(snapshot_dir))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "📰 NYT Business Sentiment"
    assert len(app.metric) == 4
    assert "120 articles" in app.markdown[0].value

    app.text_input[0].input("rally").run()
    assert not app.exception
    assert app.caption[-2].value.startswith("20 articles")  # default range: last 30 days


def test_dashboard_shows_error_without_data(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATA_URL", "http://127.0.0.1:9/unreachable")
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert app.error and "Could not load the data" in app.error[0].value
