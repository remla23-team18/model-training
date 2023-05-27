"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Model Training."""


if __name__ == "__main__":
    main(prog_name="model-training")  # pragma: no cover
