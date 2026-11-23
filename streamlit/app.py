"""NYT Business Sentiment dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from data_source import load_dashboard

import streamlit as st

POSITIVE = "#2a78d6"  # blue pole of the diverging pair
NEGATIVE = "#e34948"  # red pole
LABELS = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

st.set_page_config(page_title="NYT Business Sentiment", page_icon="📰", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Loading data…")
def get_data():
    return load_dashboard()


def _layout(fig: go.Figure, y_title: str) -> go.Figure:
    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title=y_title,
        xaxis_title=None,
    )
    fig.update_yaxes(
        gridcolor="rgba(128,128,128,0.15)", zeroline=True, zerolinecolor="rgba(128,128,128,0.5)"
    )
    fig.update_xaxes(showgrid=False)
    return fig


def index_chart(daily: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    # Reindex on calendar days so that gaps in the data break the line
    # instead of being bridged by a straight segment.
    calendar = daily.set_index("date")["sentiment_index"].asfreq("D")
    smooth = calendar.rolling(7, min_periods=3).mean()
    future = forecast.loc[forecast["is_forecast"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["sentiment_index"],
            mode="markers",
            name="Daily index",
            marker=dict(size=5, color=POSITIVE, opacity=0.25),
            hovertemplate="%{y:.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=smooth.index,
            y=smooth.values,
            mode="lines",
            name="7-day average",
            line=dict(width=2, color=POSITIVE),
            hovertemplate="%{y:.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([future["ds"], future["ds"][::-1]]),
            y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(42,120,214,0.15)",
            line=dict(width=0),
            name="Forecast interval (80%)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future["ds"],
            y=future["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(width=2, color=POSITIVE, dash="dash"),
            hovertemplate="%{y:.2f}",
        )
    )
    fig.update_yaxes(tickformat="+.2f")
    return _layout(fig, "Sentiment index (−1 to +1)")


def share_chart(daily: pd.DataFrame) -> go.Figure:
    weekly = _weekly_shares(daily)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=weekly.index,
            y=weekly["share_positive"],
            name="Positive",
            marker_color=POSITIVE,
            hovertemplate="%{y:.0%}",
        )
    )
    fig.add_trace(
        go.Bar(
            x=weekly.index,
            y=-weekly["share_negative"],
            name="Negative",
            marker_color=NEGATIVE,
            customdata=weekly["share_negative"],
            hovertemplate="%{customdata:.0%}",
        )
    )
    fig.update_layout(barmode="relative", bargap=0.15)
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, "Share of articles per week")


def _weekly_shares(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.set_index("date")
    weighted = frame[["share_positive", "share_negative"]].mul(frame["n_articles"], axis=0)
    totals = weighted.assign(n=frame["n_articles"]).resample("W").sum()
    return totals[["share_positive", "share_negative"]].div(totals["n"].clip(lower=1), axis=0)


def main() -> None:
    st.title("📰 NYT Business Sentiment")
    st.caption(
        "Daily sentiment of New York Times business coverage, scored with a multilingual "
        "DistilBERT model and forecast 30 days ahead with Prophet."
    )

    try:
        board = get_data()
    except Exception as exc:  # noqa: BLE001 - show a readable message instead of a trace
        st.error(f"Could not load the data ({exc}).")
        st.stop()

    daily, forecast, articles, meta = board.daily, board.forecast, board.articles, board.metadata
    updated = pd.to_datetime(meta.get("updated_at")) if meta.get("updated_at") else None
    st.markdown(
        f"**{len(articles):,} articles** · {meta.get('first_day')} → {meta.get('last_day')}"
        + (f" · updated {updated:%b %d, %Y}" if updated is not None else "")
        + f" · source: {board.source}"
    )

    last = daily["date"].max()
    recent = daily.loc[daily["date"] > last - pd.Timedelta(days=7), "sentiment_index"].mean()
    before = daily.loc[
        (daily["date"] <= last - pd.Timedelta(days=7))
        & (daily["date"] > last - pd.Timedelta(days=14)),
        "sentiment_index",
    ].mean()
    future = forecast.loc[forecast["is_forecast"]]
    month = daily.loc[daily["date"] > last - pd.Timedelta(days=30)]
    share_pos = (month["share_positive"] * month["n_articles"]).sum() / month["n_articles"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Sentiment, last 7 days",
        f"{recent:+.2f}",
        None if pd.isna(before) else f"{recent - before:+.2f} vs prior week",
    )
    c2.metric("Forecast in 30 days", f"{future['yhat'].iloc[-1]:+.2f}" if len(future) else "–")
    c3.metric("Positive articles, last 30 days", f"{share_pos:.0%}")
    c4.metric("Articles per day", f"{daily['n_articles'].tail(30).mean():.0f}")

    st.subheader("Sentiment index and forecast")
    st.plotly_chart(index_chart(daily, forecast), width="stretch")

    st.subheader("Positive vs negative coverage")
    st.plotly_chart(share_chart(daily), width="stretch")

    st.subheader("Explore the articles")
    left, mid, right = st.columns([2, 2, 3])
    start, end = left.date_input(
        "Date range",
        (max(articles["pub_date"].min(), last - pd.Timedelta(days=30)).date(), last.date()),
        min_value=articles["pub_date"].min().date(),
        max_value=last.date(),
    )
    labels = mid.multiselect("Sentiment", LABELS, default=LABELS)
    query = right.text_input("Keyword in headline or abstract")

    view = articles.loc[
        articles["pub_date"].dt.date.between(start, end) & articles["label"].isin(labels)
    ]
    if query:
        text = view["headline"].fillna("") + " " + view["abstract"].fillna("")
        view = view.loc[text.str.contains(query, case=False, regex=False)]
    st.dataframe(
        view.sort_values("pub_date", ascending=False)[
            ["pub_date", "headline", "label", "confidence", "web_url"]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "pub_date": st.column_config.DatetimeColumn("Published", format="MMM D, YYYY"),
            "headline": st.column_config.TextColumn("Headline", width="large"),
            "label": "Sentiment",
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "web_url": st.column_config.LinkColumn("Link", display_text="Read"),
        },
    )
    st.caption(f"{len(view):,} articles match the filters.")

    with st.expander("How it works"):
        st.markdown(
            """
- **Ingestion**: the NYT Archive API returns every article of a month; business
  sections are kept.
- **Sentiment**: `tabularisai/multilingual-sentiment-analysis` labels each
  headline + abstract on a five-point scale, mapped to −1, −0.5, 0, +0.5, +1.
- **Index**: the daily index is the mean polarity of the day's articles.
- **Forecast**: Prophet with weekly seasonality, 30 days ahead, 80% interval.
- **Orchestration**: Airflow in the local stack; a weekly GitHub Actions job
  refreshes this public snapshot.
"""
        )
    st.caption(
        "Data: The New York Times Archive API · "
        "[Source code](https://github.com/cnoret/nyt-business-sentiment)"
    )


main()
