"""Tests to check if the data is cleaned properly."""
from model_training.preprocess import clean_review

def test_clean_review() -> None:
    """It returns a cleaned review."""
    review = "This is a test."
    all_stopwords = ["this", "is", "a"]
    assert clean_review(review, all_stopwords) == "test"


def test_clean_review_no_punctuation() -> None:
    """It returns a cleaned review."""
    review = "This is a test"
    all_stopwords = ["this", "is", "a"]
    assert clean_review(review, all_stopwords) == "test"


def test_clean_review_only_punctuation() -> None:
    """It returns a cleaned review."""
    review = ".,,,."
    all_stopwords = ["this", "is", "a"]
    assert clean_review(review, all_stopwords) == ""
