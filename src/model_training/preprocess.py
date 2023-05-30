"""Reusable preprocessing script for cleaning restaurant review text data."""
import logging
import re
from pathlib import Path

import click
import nltk  # type: ignore
import pandas as pd
from click import echo
from nltk.corpus import stopwords  # type: ignore
from nltk.stem.porter import PorterStemmer  # type: ignore


logger = logging.getLogger(__name__)


def setup_stopwords() -> list[str]:
    """Download stopwords."""
    logger.debug("Downloading stopwords...")
    nltk.download("stopwords", quiet=True)
    all_stopwords: list[str] = stopwords.words("english")
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
    logger.debug("Cleaning %s...", review)
    porter_stemmer = PorterStemmer()

    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    review_words = review.split()
    review_words_striped = [
        porter_stemmer.stem(word)
        for word in review_words
        if word not in set(all_stopwords)
    ]
    review = " ".join(review_words_striped)
    logger.debug("Cleaned review: %s", review)
    return review


@click.command(name="clean")
@click.argument("review")
def clean_cli(review: str) -> None:
    """Clean a review."""
    all_stopwords = setup_stopwords()
    review = clean_review(review, all_stopwords)

    echo(f"Cleaned review: {review}")


@click.command(name="preprocess-dataset")
@click.option(
    "--dataset-path",
    default="a1_RestaurantReviews_HistoricDump.tsv",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the dataset.",
)
@click.option(
    "--output-path",
    default="a2_RestaurantReviews_Preprocessed.tsv",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to the preprocessed dataset.",
)
def preprocess_dataset(
    dataset_path: Path,
    output_path: Path,
) -> None:
    """Preprocess the dataset and save it to `output_path`."""
    dataset = pd.read_csv(dataset_path, delimiter="\t", quoting=3)
    corpus = []
    logger.debug("Cleaning and tokenizing reviews...")
    all_stopwords = setup_stopwords()
    for i in range(0, dataset["Review"].size):
        review = clean_review(dataset["Review"][i], all_stopwords)
        corpus.append(review)

    dataset["Review"] = corpus
    dataset.to_csv(output_path, sep="\t", index=False, quoting=3)
