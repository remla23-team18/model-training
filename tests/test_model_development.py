"""Test for non-determinism robustness"""

import pytest
from pathlib import Path
import json
from click.testing import CliRunner
from model_training.evaluate import evaluate_model
from model_training.train import train_model

def train(runner: CliRunner, path: Path, random_state: int = 0,preprocessed_dataset_path ="tests/resources/a2_RestaurantReviews_Preprocessed.tsv"):
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
            preprocessed_dataset_path,
            "--split-random-state",
            random_state,
            "--test-size",
            "0.2",
        ],
    )

def evaluate(runner: CliRunner, path: Path, random_state: int = 0,preprocessed_dataset_path ="tests/resources/a2_RestaurantReviews_Preprocessed.tsv"):
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
            preprocessed_dataset_path,
            "--split-random-state",
            random_state,
            "--test-size",
            "0.2",
            "--report-path",
            str(path / "classification_report.json"),
        ],
    )
    

@pytest.fixture()
def trained_model_path(runner: CliRunner, tmp_path: Path):
    train(runner, tmp_path)
    return tmp_path

# Test for non-determinism robustness.
def test_nondeterminism_robustness(runner: CliRunner,trained_model_path, tmp_path_factory):
    evaluate(runner,trained_model_path) # creates classification_report.json for original model
    original_score_path=str(trained_model_path / "classification_report.json")
    with open(original_score_path, "r") as file:
        json_data = json.load(file)
    original_score = json_data["accuracy"]
    for seed in [4,2]:
        tmp_path= tmp_path_factory.mktemp("dir"+str(seed))
        train(runner, tmp_path,random_state=seed)
        evaluate(runner,tmp_path)
        tmp_score_path=str(tmp_path / "classification_report.json")
        with open(tmp_score_path, "r") as file:
            json_data = json.load(file)
        new_score = json_data["accuracy"]
        assert abs(original_score - new_score) <=0.20

# Test for non-determinism robustness and use data slices to test model capabilities.
def test_data_slice(runner: CliRunner,trained_model_path, tmp_path_factory):
    evaluate(runner,trained_model_path) # creates classification_report.json for original model
    original_score_path=str(trained_model_path / "classification_report.json")
    with open(original_score_path, "r") as file:
        json_data = json.load(file)
    original_score = json_data["accuracy"]
    tmp_path= tmp_path_factory.mktemp("dir_slice")
    train(runner, tmp_path)
    evaluate(runner,tmp_path,preprocessed_dataset_path="tests/resources/restaurant_reviews_test_preprocessed.tsv")
    tmp_score_path=str(tmp_path / "classification_report.json")
    with open(tmp_score_path, "r") as file:
        json_data = json.load(file)
    new_score = json_data["accuracy"]
    assert abs(original_score - new_score) <=0.30