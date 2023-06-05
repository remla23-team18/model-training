"""Monitoring test for the model training pipeline."""

import psutil 
import pytest
from pathlib import Path
import json
from click.testing import CliRunner
from model_training.evaluate import evaluate_model
from model_training.train import train_model
from model_training.preprocess import preprocess_dataset_cli


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


# Monitoring ram usage for the model training pipeline.
def test_training_ram(runner: CliRunner, tmp_path: Path):
    process = psutil.Process()
    start_ram = process.memory_info().rss
    assert train(runner, tmp_path).exit_code == 0 # trains model
    end_ram = process.memory_info().rss
    ram_used = end_ram - start_ram
    assert ram_used > 1000000000 # less than 1GB of RAM used