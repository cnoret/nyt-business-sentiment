import pandas as pd
import pytest


@pytest.fixture
def archive_docs():
    """Raw documents shaped like the NYT Archive API response."""
    return [
        {
            "uri": "nyt://article/1",
            "headline": {"main": "Stocks rally as inflation cools"},
            "abstract": "Markets rose sharply.",
            "section_name": "Business Day",
            "pub_date": "2026-01-05T14:00:00+0000",
            "web_url": "https://www.nytimes.com/1",
        },
        {
            "uri": "nyt://article/2",
            "headline": {"main": "Factory closes, hundreds laid off"},
            "abstract": None,
            "snippet": "Workers fear for the region.",
            "section_name": "Business",
            "pub_date": "2026-01-05T18:30:00+0000",
            "web_url": "https://www.nytimes.com/2",
        },
        {
            "uri": "nyt://article/3",
            "headline": {"main": "Election debate recap"},
            "abstract": "Candidates clashed.",
            "section_name": "U.S.",
            "pub_date": "2026-01-06T09:00:00+0000",
            "web_url": "https://www.nytimes.com/3",
        },
        {
            "uri": "nyt://article/4",
            "headline": {"main": ""},
            "abstract": "",
            "section_name": "Business Day",
            "pub_date": "2026-01-06T09:00:00+0000",
            "web_url": "https://www.nytimes.com/4",
        },
        {
            "uri": "nyt://article/5",
            "headline": {"main": "Bad date"},
            "abstract": "Should be dropped.",
            "section_name": "Business Day",
            "pub_date": "not a date",
            "web_url": "https://www.nytimes.com/5",
        },
    ]


class FakeClassifier:
    """Deterministic stand-in for the HuggingFace pipeline."""

    RULES = (("rally", "Very Positive"), ("laid off", "Negative"), ("closes", "Negative"))

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        out = []
        for text in texts:
            label = next((lab for key, lab in self.RULES if key in text.lower()), "Neutral")
            out.append({"label": label, "score": 0.9})
        return out


@pytest.fixture
def fake_classifier():
    return FakeClassifier()


@pytest.fixture
def scored_history():
    """60 days of scored articles with a slow upward trend."""
    days = pd.date_range("2026-01-01", periods=60, freq="D")
    rows = []
    for i, day in enumerate(days):
        for j, polarity in enumerate((-0.5, 0.0, 0.5 if i > 30 else -1.0)):
            rows.append(
                {
                    "article_id": f"{i}-{j}",
                    "pub_date": day + pd.Timedelta(hours=j),
                    "polarity": polarity,
                }
            )
    return pd.DataFrame(rows)
