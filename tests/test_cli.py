"""Test cases for the __main__ module."""
from click.testing import CliRunner

from model_training import __main__


def test_cli_succeeds(runner: CliRunner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(__main__.cli)
    assert result.exit_code == 0
