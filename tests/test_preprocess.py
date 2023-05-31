"""Tests for the preprocess module."""
from pathlib import Path

from click.testing import CliRunner

from model_training import preprocess
from model_training.preprocess import preprocess_dataset_cli


def test_clean(runner: CliRunner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(preprocess.clean_cli, ["This is a test."])
    assert result.exit_code == 0
    assert result.output == "Cleaned review: test\n"


def test_preprocess_entire_dataset(runner: CliRunner, tmp_path: Path) -> None:
    """It exits with a status code of zero."""
    # Create a dummy dataset
    (tmp_path / "test.tsv").write_text("Review\tLiked\nThis is a test\t1\n")
    result = runner.invoke(
        preprocess_dataset_cli,
        [
            "--dataset-path",
            str(tmp_path / "test.tsv"),
            "--output-path",
            str(tmp_path / "test_preprocessed.tsv"),
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "test_preprocessed.tsv").exists()
    assert (
        tmp_path / "test_preprocessed.tsv"
    ).read_text() == "Review\tLiked\ntest\t1\n"
