"""Cleaning and filtering of raw articles before sentiment scoring."""

from __future__ import annotations

import pandas as pd

BUSINESS_SECTIONS = frozenset({"Business", "Business Day"})


def clean_articles(
    articles: pd.DataFrame, sections: frozenset[str] = BUSINESS_SECTIONS
) -> pd.DataFrame:
    """Keep business articles with a valid date and a non-empty text.

    Adds a ``text`` column (headline + abstract) used as model input and a
    naive-UTC ``pub_date``. Duplicates are removed on ``article_id``.
    """
    df = articles.loc[articles["section"].isin(sections)].copy()

    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["pub_date"])
    df["pub_date"] = df["pub_date"].dt.tz_localize(None)

    headline = df["headline"].fillna("").astype(str).str.strip()
    abstract = df["abstract"].fillna("").astype(str).str.strip()
    df["text"] = (headline + ". " + abstract).str.strip(" .")
    df = df.loc[df["text"].str.len() > 0]

    df = df.drop_duplicates(subset="article_id", keep="last")
    return df.sort_values("pub_date").reset_index(drop=True)
