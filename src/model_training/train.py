"""Training pipeline for the sentiment analysis model."""
import logging
import pickle
from pathlib import Path

import click
import joblib  # type: ignore
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.naive_bayes import GaussianNB  # type: ignore

from .preprocess import clean_review
from .preprocess import setup_stopwords


logger = logging.getLogger(__name__)


@click.command(name="train")
@click.option(
    "--output-dir",
    default="models",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    help="Directory where the model should be saved.",
)
@click.option(
    "--count-vectorizer-artifact-name",
    default="c1_BoW_Sentiment_Model.pkl",
    type=click.Path(path_type=Path),
    help="Name of the count vectorizer artifact.",
)
@click.option(
    "--classifier-artifact-name",
    default="c2_Classifier_Sentiment_Model",
    type=click.Path(path_type=Path),
    help="Name of the classifier artifact.",
)
@click.option(
    "--dataset-path",
    default="a1_RestaurantReviews_HistoricDump.tsv",
    type=click.Path(path_type=Path, dir_okay=False, readable=True),
    help="Path to the (unprocessed) dataset.",
)
@click.option(
    "--preprocessed-dataset-path",
    default=None,
    type=click.Path(path_type=Path, dir_okay=False, readable=True),
    help="Path to the preprocessed dataset.",
)
@click.option(
    "--split-random-state",
    default=0,
    help="Random state for the train/test split.",
)
@click.option(
    "--test-size",
    default=0.2,
    help="Size of the test set.",
)
def train_model(
    output_dir: Path,
    count_vectorizer_artifact_name: Path,
    classifier_artifact_name: Path,
    dataset_path: Path,
    preprocessed_dataset_path: Path | None,
    split_random_state: int,
    test_size: float,
) -> None:
    """Train the sentiment analysis model."""
    corpus: list[str] = []
    if preprocessed_dataset_path is None:
        logger.debug("Preprocessing the dataset...")
        dataset = pd.read_csv(dataset_path, delimiter="\t", quoting=3)

        logger.debug("Cleaning and tokenizing reviews...")
        all_stopwords = setup_stopwords()
        for i in range(0, dataset["Review"].size):
            review = clean_review(dataset["Review"][i], all_stopwords)
            corpus.append(review)
    else:
        logger.debug("Loading the preprocessed dataset...")
        dataset = pd.read_csv(
            preprocessed_dataset_path, delimiter="\t", quoting=3, keep_default_na=False
        )
        corpus = dataset["Review"].tolist()
        logger.debug("Corpus size: %d", len(corpus))

    logger.debug("Creating the Bag of Words model...")
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    logger.info("Splitting the dataset into the Training set and Test set...")
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=split_random_state
    )
    classifier = GaussianNB()
    logger.info("Fitting Naive Bayes to the Training set...")
    classifier.fit(X_train, y_train)

    logger.info("Saving the model...")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    logger.debug("Saving the count vectorizer to later use in prediction")
    pickle.dump(cv, open(output_dir_path / count_vectorizer_artifact_name, "wb"))

    logger.debug("Saving the classifier to later use in prediction")
    joblib.dump(classifier, output_dir_path / classifier_artifact_name)
