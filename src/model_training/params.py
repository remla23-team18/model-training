"""Functions to load the parameters from the parameters file."""
import logging
from pathlib import Path

import yaml  # type: ignore


logger = logging.getLogger(__name__)


def load_params(params_path: Path) -> tuple[int, float]:
    """Load the parameters from the parameters file."""
    logger.debug("Loading the parameters...")
    with open(params_path) as params_file:
        params = yaml.safe_load(params_file)
    split_random_state = params["split_random_state"]
    test_size = params["test_size"]
    return split_random_state, test_size
