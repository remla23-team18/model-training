"""Command-line interface."""
import logging

import click

from .preprocess import clean_cli
from .train import train_model


logger = logging.basicConfig(level=logging.INFO)


@click.group()
@click.version_option()
def cli() -> None:
    """Command line tool to train a sentiment analysis on restaurant reviews."""
    pass  # pragma: no cover


cli.add_command(train_model)
cli.add_command(clean_cli)


if __name__ == "__main__":
    cli(prog_name="model-training")  # pragma: no cover
