"""Tests for the train module."""
from pathlib import Path

from click.testing import CliRunner

from model_training import train


def test_train(runner: CliRunner, tmp_path: Path) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(
        train.train_model,
        [
            "--output-dir",
            str(tmp_path / "models"),
            "--dataset-path",
            "tests/resources/restaurant_reviews_test.tsv",
        ],
    )
    assert result.exit_code == 0


def test_train_preprocessed(runner: CliRunner, tmp_path: Path) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(
        train.train_model,
        [
            "--output-dir",
            str(tmp_path / "models"),
            "--dataset-path",
            "tests/resources/restaurant_reviews_test.tsv",
            "--preprocessed-dataset-path",
            "tests/resources/restaurant_reviews_test_preprocessed.tsv",
        ],
    )
    assert result.exit_code == 0


def test_train_preprocessed_with_params(runner: CliRunner, tmp_path: Path) -> None:
    """It exits with a status code of zero."""
    params_path = tmp_path / "params.yaml"
    params_path.write_text("split_random_state: 0\ntest_size: 0.2\n")

    result = runner.invoke(
        train.train_model,
        [
            "--output-dir",
            str(tmp_path / "models"),
            "--dataset-path",
            "tests/resources/restaurant_reviews_test.tsv",
            "--preprocessed-dataset-path",
            "tests/resources/restaurant_reviews_test_preprocessed.tsv",
            "--params-path",
            str(params_path),
        ],
    )
    assert result.exit_code == 0
