from nyt_sentiment.nyt_api import docs_to_frame
from nyt_sentiment.preprocess import clean_articles


def test_clean_articles_keeps_valid_business_articles(archive_docs):
    cleaned = clean_articles(docs_to_frame(archive_docs))
    # Non-business, empty-text and bad-date articles are dropped.
    assert cleaned["article_id"].tolist() == ["nyt://article/1", "nyt://article/2"]
    assert cleaned.loc[0, "text"] == "Stocks rally as inflation cools. Markets rose sharply"
    assert cleaned["pub_date"].dt.tz is None


def test_clean_articles_removes_duplicates(archive_docs):
    frame = docs_to_frame(archive_docs + archive_docs[:1])
    cleaned = clean_articles(frame)
    assert cleaned["article_id"].is_unique
