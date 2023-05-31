"""Test the parameter loading function."""

from pathlib import Path

from model_training.params import load_params


def test_load_params(tmp_path: Path) -> None:
    """It returns the correct parameters."""
    params_path = tmp_path / "params.yaml"
    params_path.write_text("split_random_state: 0\ntest_size: 0.2\n")
    split_random_state, test_size = load_params(params_path)
    assert split_random_state == 0
    assert test_size == 0.2
