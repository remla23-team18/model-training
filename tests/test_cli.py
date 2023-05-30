"""Test cases for the __main__ module."""
import pytest
from click.testing import CliRunner

from model_training import __main__


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
    result = runner.invoke(__main__.clean_cli, ["This is a test."])
    assert result.exit_code == 0
    assert result.output == "Cleaned review: test\n"


def test_train(runner: CliRunner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(__main__.train_model)
    assert result.exit_code == 0
