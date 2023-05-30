"""Reusable preprocessing script for cleaning restaurant review text data."""
import logging
import re

import click
import nltk
from click import echo
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


logger = logging.getLogger(__name__)


def setup_stopwords() -> list[str]:
    """Download stopwords."""
    logger.debug("Downloading stopwords...")
    nltk.download("stopwords", quiet=True)
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")

    return all_stopwords


def clean_review(review: str, all_stopwords: list[str]) -> str:
    """Clean review text.

    Parameters
    ----------
    review : str
        Review text to be cleaned.
    all_stopwords : list[str]
        List of stopwords to remove from the review text.

    Returns
    -------
    review : str
        Cleaned review text.
    """
    logger.debug(f"Cleaning {review}...")
    ps = PorterStemmer()

    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = " ".join(review)
    logger.debug(f"Cleaned review: {review}")
    return review


@click.command(name="clean")
@click.argument("review")
def clean_cli(review: str) -> None:
    """Clean a review."""
    stopwords = setup_stopwords()
    review = clean_review(review, stopwords)

    echo(f"Cleaned review: {review}")
