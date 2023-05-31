"""Test cases for the __main__ module."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from model_training import __main__
from model_training import preprocess
from model_training import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_cli_succeeds(runner: CliRunner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(__main__.cli)
    assert result.exit_code == 0


def test_clean(runner: CliRunner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(preprocess.clean_cli, ["This is a test."])
    assert result.exit_code == 0
    assert result.output == "Cleaned review: test\n"


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
