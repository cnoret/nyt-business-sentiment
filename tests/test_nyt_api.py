import pandas as pd
import pytest

from nyt_sentiment import nyt_api


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def get(self, url, params=None, timeout=None):
        self.requests.append((url, params))
        return self.responses.pop(0)


def test_fetch_archive_month_returns_docs(archive_docs):
    session = FakeSession([FakeResponse(200, {"response": {"docs": archive_docs}})])
    docs = nyt_api.fetch_archive_month(2026, 1, "key", session=session)
    assert docs == archive_docs
    url, params = session.requests[0]
    assert url.endswith("/archive/v1/2026/1.json")
    assert params == {"api-key": "key"}


def test_fetch_archive_month_raises_on_http_error():
    session = FakeSession([FakeResponse(401)])
    with pytest.raises(nyt_api.NYTAPIError, match="401"):
        nyt_api.fetch_archive_month(2026, 1, "bad", session=session)


def test_fetch_archive_month_requires_key():
    with pytest.raises(nyt_api.NYTAPIError, match="NYT_API_KEY"):
        nyt_api.fetch_archive_month(2026, 1, "")


def test_fetch_archive_month_retries_on_rate_limit(monkeypatch, archive_docs):
    monkeypatch.setattr(nyt_api.time, "sleep", lambda _: None)
    session = FakeSession(
        [FakeResponse(429), FakeResponse(200, {"response": {"docs": archive_docs}})]
    )
    assert nyt_api.fetch_archive_month(2026, 1, "key", session=session) == archive_docs
    assert len(session.requests) == 2


def test_docs_to_frame_flattens_documents(archive_docs):
    frame = nyt_api.docs_to_frame(archive_docs)
    assert list(frame.columns) == nyt_api.ARTICLE_COLUMNS
    assert frame.loc[0, "headline"] == "Stocks rally as inflation cools"
    # Missing abstract falls back to the snippet.
    assert frame.loc[1, "abstract"] == "Workers fear for the region."
    assert frame["pub_date"].isna().sum() == 1


def test_months_between_spans_year_boundary():
    months = nyt_api.months_between(pd.Timestamp("2025-11-20"), pd.Timestamp("2026-02-01"))
    assert months == [(2025, 11), (2025, 12), (2026, 1), (2026, 2)]


def test_fetch_months_pauses_between_calls(monkeypatch):
    sleeps = []
    monkeypatch.setattr(nyt_api.time, "sleep", sleeps.append)
    docs = nyt_api.fetch_months(
        [(2026, 1), (2026, 2), (2026, 3)], "key", fetch=lambda y, m, k: [{"m": m}], pause=12
    )
    assert docs == [{"m": 1}, {"m": 2}, {"m": 3}]
    assert sleeps == [12, 12]
