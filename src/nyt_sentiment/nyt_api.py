"""Client for the New York Times Archive API.

The Archive API returns every article published in a given month, which makes
it the right source for building a daily time series (the Newswire endpoint
only exposes the latest items).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import pandas as pd
import requests

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"

ARTICLE_COLUMNS = [
    "article_id",
    "headline",
    "abstract",
    "section",
    "pub_date",
    "web_url",
]


class NYTAPIError(RuntimeError):
    """Raised when the NYT API answers with an error."""


def fetch_archive_month(
    year: int,
    month: int,
    api_key: str,
    session: requests.Session | None = None,
    timeout: float = 60.0,
    retries: int = 5,
    backoff: float = 30.0,
) -> list[dict]:
    """Return the raw documents published during ``year``/``month``."""
    if not api_key:
        raise NYTAPIError("NYT_API_KEY is not set.")

    http = session or requests.Session()
    url = ARCHIVE_URL.format(year=year, month=month)
    for attempt in range(1, retries + 1):
        response = http.get(url, params={"api-key": api_key}, timeout=timeout)
        if response.status_code == 429 and attempt < retries:
            # The API allows 5 requests per minute and 500 per day: back off and retry.
            time.sleep(backoff * attempt)
            continue
        if response.status_code != 200:
            raise NYTAPIError(
                f"NYT Archive API returned HTTP {response.status_code} for {year}-{month:02d}"
            )
        return response.json().get("response", {}).get("docs", [])
    raise NYTAPIError(f"NYT Archive API rate limit hit for {year}-{month:02d}")


def fetch_months(
    months: list[tuple[int, int]],
    api_key: str,
    fetch=fetch_archive_month,
    pause: float = 12.0,
) -> list[dict]:
    """Fetch several months, pausing between calls to respect the rate limit."""
    docs: list[dict] = []
    for index, (year, month) in enumerate(months):
        if index and pause:
            time.sleep(pause)
        month_docs = fetch(year, month, api_key)
        log.info("Fetched %d documents for %d-%02d", len(month_docs), year, month)
        docs.extend(month_docs)
    return docs


def docs_to_frame(docs: Iterable[dict]) -> pd.DataFrame:
    """Flatten raw Archive API documents into a tidy articles table."""
    rows = []
    for doc in docs:
        headline = doc.get("headline") or {}
        rows.append(
            {
                "article_id": doc.get("uri") or doc.get("_id"),
                "headline": headline.get("main") if isinstance(headline, dict) else headline,
                "abstract": doc.get("abstract") or doc.get("snippet"),
                "section": doc.get("section_name") or doc.get("news_desk"),
                "pub_date": doc.get("pub_date"),
                "web_url": doc.get("web_url"),
            }
        )
    frame = pd.DataFrame(rows, columns=ARTICLE_COLUMNS)
    frame["pub_date"] = pd.to_datetime(frame["pub_date"], errors="coerce", utc=True)
    return frame


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    """List the (year, month) pairs covering ``start``..``end`` inclusive."""
    periods = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    return [(p.year, p.month) for p in periods]
