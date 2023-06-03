"""Integration test for the full ML pipeline"""

import pytest
from pathlib import Path
import json
from click.testing import CliRunner
from model_training.evaluate import evaluate_model
from model_training.train import train_model
from model_training.preprocess import preprocess_dataset_cli

def preprocess(runner: CliRunner, tmp_path: Path):
    return runner.invoke(
        preprocess_dataset_cli,
        [
            "--dataset-path",
            "tests/resources/a1_RestaurantReviews_HistoricDump.tsv",
            "--output-path",
            str(tmp_path / "test_preprocessed.tsv"),
        ],
    )

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
    

# Integration test for the full ML pipeline
def test_pipeline(runner: CliRunner, tmp_path: Path):
    assert preprocess(runner, tmp_path).exit_code == 0 # preprocesses data
    assert (tmp_path / "test_preprocessed.tsv").exists()
    assert train(runner, tmp_path,preprocessed_dataset_path=str(tmp_path / "test_preprocessed.tsv")).exit_code == 0 # trains model
    assert (tmp_path / "models").exists()
    assert evaluate(runner,tmp_path,preprocessed_dataset_path=str(tmp_path / "test_preprocessed.tsv")).exit_code == 0 # evaluates model
    assert (tmp_path / "classification_report.json").exists()
