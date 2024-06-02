"""Console script for action_rules."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("action-rules")
    click.echo("=" * len("action-rules"))
    click.echo("The package for action rules mining using Action-Apriori (Apriori Modified for Action Rules Mining).")


if __name__ == "__main__":
    main()  # pragma: no cover
