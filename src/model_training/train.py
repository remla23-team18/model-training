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

from .params import load_params
from .preprocess import make_preprocessed_dataset


logger = logging.getLogger(__name__)


def _create_corpus(dataset_path: Path) -> tuple[list[str], list[int]]:
    logger.debug("Preprocessing the dataset...")
    dataset = make_preprocessed_dataset(dataset_path)
    corpus = dataset["Review"].tolist()
    labels = dataset["Liked"].tolist()
    return corpus, labels


def _load_existing_corpus(
    preprocessed_dataset_path: Path,
) -> tuple[list[str], list[int]]:
    logger.debug("Loading the preprocessed dataset...")
    dataset = pd.read_csv(
        preprocessed_dataset_path,
        delimiter="\t",
        quoting=3,
        keep_default_na=False,
        dtype={"Liked": int, "Review": str},
    )
    dataset = dataset[["Review", "Liked"]]
    corpus = dataset["Review"].tolist()
    labels = dataset["Liked"].tolist()
    logger.debug("Corpus size: %d", len(corpus))

    return corpus, labels


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
@click.option(
    "--params-path",
    type=click.Path(path_type=Path, dir_okay=False, readable=True),
    help="Path to the parameters file.",
)
def train_model(
    output_dir: Path,
    count_vectorizer_artifact_name: Path,
    classifier_artifact_name: Path,
    dataset_path: Path,
    preprocessed_dataset_path: Path | None,
    split_random_state: int,
    test_size: float,
    params_path: Path | None,
) -> None:
    """Train the sentiment analysis model."""
    if params_path is not None:
        split_random_state, test_size = load_params(params_path)
    corpus: list[str] = []
    y: list[int] = []
    if preprocessed_dataset_path is None:
        corpus, y = _create_corpus(dataset_path)
    else:
        corpus, y = _load_existing_corpus(preprocessed_dataset_path)

    logger.debug("Creating the Bag of Words model...")
    count_vectorizer = CountVectorizer(max_features=1420)

    X = count_vectorizer.fit_transform(corpus).toarray()

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
    with open(output_dir_path / count_vectorizer_artifact_name, "wb") as output_file:
        pickle.dump(count_vectorizer, output_file)

    logger.debug("Saving the classifier to later use in prediction")
    joblib.dump(classifier, output_dir_path / classifier_artifact_name)
