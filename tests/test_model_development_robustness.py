"""Test for non-determinism robustness."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from click.testing import Result
from pytest import TempPathFactory

from model_training.evaluate import evaluate_model
from model_training.train import train_model


def _train(
    runner: CliRunner,
    path: Path,
    random_state: int = 0,
    preprocessed_dataset_path: Path | None = None,
) -> Result:
    if preprocessed_dataset_path is None:
        preprocessed_dataset_path = Path(
            "tests/resources/a2_RestaurantReviews_Preprocessed.tsv"
        )
    count_vectorizer_artifact_name = "cv.pkl"
    classifier_artifact_name = "clf"

    # Train a model on minimal test data
    models_path = path / "models"
    models_path.mkdir()
    return runner.invoke(
        train_model,
        [
            "--output-dir",
            str(models_path),
            "--count-vectorizer-artifact-name",
            count_vectorizer_artifact_name,
            "--classifier-artifact-name",
            classifier_artifact_name,
            "--preprocessed-dataset-path",
            str(preprocessed_dataset_path),
            "--split-random-state",
            str(random_state),
            "--test-size",
            "0.2",
        ],
    )


def _evaluate(
    runner: CliRunner,
    path: Path,
    random_state: int = 0,
    preprocessed_dataset_path: Path | None = None,
) -> Result:
    if preprocessed_dataset_path is None:
        preprocessed_dataset_path = Path(
            "tests/resources/a2_RestaurantReviews_Preprocessed.tsv"
        )
    count_vectorizer_artifact_name = "cv.pkl"
    classifier_artifact_name = "clf"
    models_path = path / "models"
    return runner.invoke(
        evaluate_model,
        [
            "--model-dir",
            str(models_path),
            "--count-vectorizer-artifact-name",
            count_vectorizer_artifact_name,
            "--classifier-artifact-name",
            classifier_artifact_name,
            "--preprocessed-dataset-path",
            str(preprocessed_dataset_path),
            "--split-random-state",
            str(random_state),
            "--test-size",
            "0.2",
            "--report-path",
            str(path / "classification_report.json"),
        ],
    )


@pytest.fixture(name="trained_model_path")
def trained_model_path_fixture(runner: CliRunner, tmp_path: Path) -> Path:
    """Train a model on minimal test data.

    Return the temp path to the trained model.
    """
    _train(runner, tmp_path)
    return tmp_path


def test_nondeterminism_robustness(
    runner: CliRunner, trained_model_path: Path, tmp_path_factory: TempPathFactory
) -> None:
    """Test for non-determinism robustness.

    Trains two models with different random seeds and comparing their accuracy scores.

    The test passes if the difference between the two scores is less than 0.20.
    """
    _evaluate(
        runner, trained_model_path
    )  # creates classification_report.json for original model
    original_score_path = str(trained_model_path / "classification_report.json")
    with open(original_score_path) as file:
        json_data = json.load(file)
    original_score = json_data["accuracy"]
    for seed in [4, 2]:
        tmp_path = tmp_path_factory.mktemp("dir" + str(seed))
        _train(runner, tmp_path, random_state=seed)
        _evaluate(runner, tmp_path)
        tmp_score_path = str(tmp_path / "classification_report.json")
        with open(tmp_score_path) as file:
            json_data = json.load(file)
        new_score = json_data["accuracy"]
        assert abs(original_score - new_score) <= 0.20


def test_data_slice(
    runner: CliRunner, trained_model_path: Path, tmp_path_factory: TempPathFactory
) -> None:
    """Test for non-determinism robustness using a positive data slice."""
    _evaluate(
        runner, trained_model_path
    )  # creates classification_report.json for original model
    original_score_path = str(trained_model_path / "classification_report.json")
    with open(original_score_path) as file:
        json_data = json.load(file)
    original_score = json_data["accuracy"]
    tmp_path = tmp_path_factory.mktemp("dir_slice")
    _train(runner, tmp_path)
    _evaluate(
        runner,
        tmp_path,
        preprocessed_dataset_path=Path(
            "tests/resources/preprocessed_positive_only.tsv"
        ),
    )
    tmp_score_path = str(tmp_path / "classification_report.json")
    with open(tmp_score_path) as file:
        json_data = json.load(file)
    new_score = json_data["accuracy"]
    assert abs(original_score - new_score) <= 0.30
