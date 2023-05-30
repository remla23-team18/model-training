"""Command-line interface."""
import logging

import click

from .evaluate import evaluate_model
from .preprocess import clean_cli
from .preprocess import preprocess_dataset
from .train import train_model


logging.basicConfig(level=logging.INFO)


@click.group()
@click.version_option()
def cli() -> None:
    """Command line tool to train a sentiment analysis on restaurant reviews."""


cli.add_command(train_model)
cli.add_command(clean_cli)
cli.add_command(preprocess_dataset)
cli.add_command(evaluate_model)


if __name__ == "__main__":
    cli(prog_name="model-training")  # pragma: no cover
