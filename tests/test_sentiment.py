import pytest

from nyt_sentiment.nyt_api import docs_to_frame
from nyt_sentiment.preprocess import clean_articles
from nyt_sentiment.sentiment import LABEL_POLARITY, polarity, score_articles, score_texts


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Very Negative", -1.0),
        ("Negative", -0.5),
        ("Neutral", 0.0),
        ("Positive", 0.5),
        ("Very Positive", 1.0),
    ],
)
def test_polarity_mapping(label, expected):
    assert polarity(label) == expected


def test_polarity_is_symmetric():
    values = sorted(LABEL_POLARITY.values())
    assert values == [-v for v in reversed(values)]


def test_polarity_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unknown sentiment label"):
        polarity("POSITIVE")


def test_score_texts_batches_calls(fake_classifier):
    texts = [f"text {i}" for i in range(5)]
    scores = score_texts(texts, fake_classifier, batch_size=2)
    assert [len(c) for c in fake_classifier.calls] == [2, 2, 1]
    assert list(scores.columns) == ["label", "confidence", "polarity"]
    assert len(scores) == 5


def test_score_articles_appends_sentiment(archive_docs, fake_classifier):
    articles = clean_articles(docs_to_frame(archive_docs))
    scored = score_articles(articles, fake_classifier)
    assert scored["label"].tolist() == ["Very Positive", "Negative"]
    assert scored["polarity"].tolist() == [1.0, -0.5]
    assert scored["article_id"].tolist() == articles["article_id"].tolist()
