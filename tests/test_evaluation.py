"""Test the evaluation command."""

from pathlib import Path

from click.testing import CliRunner

from model_training.evaluate import evaluate_model
from model_training.train import train_model


def test_evaluate(runner: CliRunner, tmp_path: Path) -> None:
    """It exits with a status code of zero."""
    count_vectorizer_artifact_name = "cv.pkl"
    classifier_artifact_name = "clf"

    # Train a model on minimal test data
    models_path = tmp_path / "models"
    models_path.mkdir()
    runner.invoke(
        train_model,
        [
            "--output-dir",
            str(models_path),
            "--count-vectorizer-artifact-name",
            count_vectorizer_artifact_name,
            "--classifier-artifact-name",
            classifier_artifact_name,
            "--preprocessed-dataset-path",
            "tests/resources/restaurant_reviews_test_preprocessed.tsv",
            "--split-random-state",
            "0",
            "--test-size",
            "0.2",
        ],
    )

    result = runner.invoke(
        evaluate_model,
        [
            "--model-dir",
            str(models_path),
            "--count-vectorizer-artifact-name",
            count_vectorizer_artifact_name,
            "--classifier-artifact-name",
            classifier_artifact_name,
            "--preprocessed-dataset-path",
            "tests/resources/restaurant_reviews_test_preprocessed.tsv",
            "--split-random-state",
            "0",
            "--test-size",
            "0.2",
            "--report-path",
            str(tmp_path / "classification_report.json"),
        ],
    )
    assert result.exit_code == 0


def test_evaluate_with_params(runner: CliRunner, tmp_path: Path) -> None:
    """It exits with a status code of zero."""
    params_path = tmp_path / "params.yaml"
    params_path.write_text("split_random_state: 0\ntest_size: 0.2\n")
    count_vectorizer_artifact_name = "cv.pkl"
    classifier_artifact_name = "clf"

    # Train a model on minimal test data
    models_path = tmp_path / "models"
    models_path.mkdir()
    runner.invoke(
        train_model,
        [
            "--output-dir",
            str(models_path),
            "--count-vectorizer-artifact-name",
            count_vectorizer_artifact_name,
            "--classifier-artifact-name",
            classifier_artifact_name,
            "--preprocessed-dataset-path",
            "tests/resources/restaurant_reviews_test_preprocessed.tsv",
            "--split-random-state",
            "0",
            "--test-size",
            "0.2",
        ],
    )

    result = runner.invoke(
        evaluate_model,
        [
            "--model-dir",
            str(models_path),
            "--count-vectorizer-artifact-name",
            count_vectorizer_artifact_name,
            "--classifier-artifact-name",
            classifier_artifact_name,
            "--preprocessed-dataset-path",
            "tests/resources/restaurant_reviews_test_preprocessed.tsv",
            "--split-random-state",
            "0",
            "--test-size",
            "0.2",
            "--report-path",
            str(tmp_path / "classification_report.json"),
            "--params-path",
            str(params_path),
        ],
    )
    assert result.exit_code == 0
