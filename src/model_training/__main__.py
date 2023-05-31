"""Command-line interface."""
import logging

import click

from .evaluate import evaluate_model
from .preprocess import clean_cli
from .preprocess import preprocess_dataset_cli
from .train import train_model


logging.basicConfig(level=logging.INFO)


@click.group()
@click.version_option()
def cli() -> None:
    """Command line tool to train a sentiment analysis on restaurant reviews."""


cli.add_command(train_model)
cli.add_command(clean_cli)
cli.add_command(preprocess_dataset_cli)
cli.add_command(evaluate_model)


if __name__ == "__main__":
    # pylint: disable=unexpected-keyword-arg
    cli(prog_name="model-training")  # pragma: no cover
    # pylint: enable=unexpected-keyword-arg
