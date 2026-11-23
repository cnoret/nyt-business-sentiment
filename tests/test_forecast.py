import pandas as pd
import pytest

from nyt_sentiment.forecast import daily_sentiment, forecast_index


def test_daily_sentiment_aggregates_per_day():
    scored = pd.DataFrame(
        {
            "pub_date": pd.to_datetime(
                ["2026-01-01 08:00", "2026-01-01 17:00", "2026-01-01 20:00", "2026-01-02 09:00"]
            ),
            "polarity": [1.0, -0.5, 0.0, 0.5],
        }
    )
    daily = daily_sentiment(scored)
    first = daily.iloc[0]
    assert len(daily) == 2
    assert first["sentiment_index"] == pytest.approx(1 / 6)
    assert first["share_positive"] == pytest.approx(1 / 3)
    assert first["share_negative"] == pytest.approx(1 / 3)
    assert first["n_articles"] == 3


def test_forecast_index_extends_horizon(scored_history):
    daily = daily_sentiment(scored_history)
    forecast = forecast_index(daily, horizon_days=10)
    assert len(forecast) == len(daily) + 10
    assert forecast["is_forecast"].sum() == 10
    assert (forecast["yhat_lower"] <= forecast["yhat_upper"]).all()


def test_forecast_index_requires_history(scored_history):
    daily = daily_sentiment(scored_history).head(5)
    with pytest.raises(ValueError, match="at least"):
        forecast_index(daily)
