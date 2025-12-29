"""Command line interface for DODO."""

from .run_coupled_optimization import main


def cli() -> None:
    """Entry point for dodo-optimize."""
    main()


if __name__ == "__main__":
    cli()
