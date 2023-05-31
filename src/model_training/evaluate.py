"""Report metrics for the trained sentiment analysis model."""
import json
import logging
from pathlib import Path

import click
import joblib  # type: ignore
import pandas as pd
from sklearn.metrics import classification_report  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


logger = logging.getLogger(__name__)


@click.command(name="evaluate")
@click.option(
    "--model-dir",
    default="models",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
    help="Directory where the model should be loaded from.",
)
@click.option(
    "--count-vectorizer-artifact-name",
    default="c1_BoW_Sentiment_Model.pkl",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Name of the count vectorizer artifact.",
)
@click.option(
    "--classifier-artifact-name",
    default="c2_Classifier_Sentiment_Model",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Name of the classifier artifact.",
)
@click.option(
    "--preprocessed-dataset-path",
    default="a2_RestaurantReviews_Preprocessed.tsv",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to the dataset.",
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
    "--report-path",
    default="classification_report.json",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to the classification report.",
)
def evaluate_model(
    model_dir: Path,
    count_vectorizer_artifact_name: Path,
    classifier_artifact_name: Path,
    preprocessed_dataset_path: Path,
    split_random_state: int,
    test_size: float,
    report_path: Path,
) -> None:
    """Evaluate the sentiment analysis model."""
    logger.info("Evaluating the model...")
    logger.debug("Loading the model...")
    model_dir_path = Path(model_dir)
    count_vectorizer = joblib.load(model_dir_path / count_vectorizer_artifact_name)
    classifier = joblib.load(model_dir_path / classifier_artifact_name)

    logger.debug("Loading the dataset...")
    dataset = pd.read_csv(preprocessed_dataset_path, delimiter="\t", quoting=3)
    logger.debug("Splitting the dataset...")
    _, X_test, _, y_test = train_test_split(
        dataset["Review"],
        dataset["Liked"],
        test_size=test_size,
        random_state=split_random_state,
    )

    logger.debug("Vectorizing the test set...")
    X_test = count_vectorizer.transform(X_test).toarray()
    logger.debug("Predicting the test set...")
    y_pred = classifier.predict(X_test)
    logger.debug("Reporting metrics...")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    logger.info("Classification report: %s", report_str)
    click.echo(f"Classification report: {report_str}")

    logger.debug("Saving the classification report...")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report))
