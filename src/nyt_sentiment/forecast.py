"""Daily sentiment index and its Prophet forecast."""

from __future__ import annotations

import pandas as pd


def daily_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored articles into one row per day.

    Columns:
    - ``sentiment_index``: mean polarity of the day's articles, in [-1, 1]
    - ``share_positive`` / ``share_negative``: fraction of articles on each side
    - ``n_articles``: number of articles scored that day
    """
    df = scored.assign(date=pd.to_datetime(scored["pub_date"]).dt.normalize())
    daily = df.groupby("date").agg(
        sentiment_index=("polarity", "mean"),
        share_positive=("polarity", lambda s: (s > 0).mean()),
        share_negative=("polarity", lambda s: (s < 0).mean()),
        n_articles=("polarity", "size"),
    )
    return daily.reset_index()


def forecast_index(daily: pd.DataFrame, horizon_days: int = 30, min_days: int = 14) -> pd.DataFrame:
    """Fit Prophet on the daily index and forecast ``horizon_days`` ahead.

    Returns ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper`` and ``is_forecast``.
    """
    from prophet import Prophet

    history = daily.rename(columns={"date": "ds", "sentiment_index": "y"})[["ds", "y"]]
    history = history.dropna()
    if len(history) < min_days:
        raise ValueError(
            f"Need at least {min_days} days of history to forecast, got {len(history)}."
        )

    model = Prophet(weekly_seasonality=True, yearly_seasonality=False)
    model.fit(history)
    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast["is_forecast"] = forecast["ds"] > history["ds"].max()
    return forecast
