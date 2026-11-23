"""Sentiment scoring with a HuggingFace text-classification model."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from nyt_sentiment.config import DEFAULT_MODEL

# The default model predicts five classes. Each one is mapped to a polarity in
# [-1, 1] so that articles can be aggregated into a single daily index.
LABEL_POLARITY: dict[str, float] = {
    "Very Negative": -1.0,
    "Negative": -0.5,
    "Neutral": 0.0,
    "Positive": 0.5,
    "Very Positive": 1.0,
}

Classifier = Callable[[list[str]], list[dict]]


def load_classifier(model_name: str = DEFAULT_MODEL) -> Classifier:
    """Build a CPU text-classification pipeline (imported lazily: heavy)."""
    from transformers import pipeline

    return pipeline(
        task="text-classification",
        model=model_name,
        device=-1,
        truncation=True,
        max_length=512,
    )


def polarity(label: str) -> float:
    """Return the polarity of a model label, failing loudly on unknown ones."""
    try:
        return LABEL_POLARITY[label]
    except KeyError as exc:
        raise ValueError(f"Unknown sentiment label: {label!r}") from exc


def score_texts(texts: Sequence[str], classifier: Classifier, batch_size: int = 32) -> pd.DataFrame:
    """Score ``texts`` and return ``label``, ``confidence`` and ``polarity``."""
    predictions: list[dict] = []
    for start in range(0, len(texts), batch_size):
        predictions.extend(classifier(list(texts[start : start + batch_size])))

    scores = pd.DataFrame(predictions, columns=["label", "score"]).rename(
        columns={"score": "confidence"}
    )
    scores["polarity"] = scores["label"].map(polarity)
    return scores


def score_articles(
    articles: pd.DataFrame, classifier: Classifier, batch_size: int = 32
) -> pd.DataFrame:
    """Append sentiment columns to a cleaned articles table."""
    scores = score_texts(articles["text"].tolist(), classifier, batch_size)
    return pd.concat([articles.reset_index(drop=True), scores], axis=1)
