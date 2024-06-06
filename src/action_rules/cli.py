"""Console script for action_rules."""

import click
import pandas as pd


@click.command()
def main():
    """Entrypoint for console script."""
    click.echo("action-rules")
    click.echo("=" * len("action-rules"))
    click.echo("The package for action rules mining using Action-Apriori (Apriori Modified for Action Rules Mining).")

    click.option('--csv_path', prompt='CSV Path', help='Dataset where the first row is the header. A comma is used as a separator.')
    data = pd.DataFrame.read_csv(csv_path)
    cols = str(list(data.columns)[:20])

    click.option('--min_stable_attributes', prompt='Min stable attributes', help='The person to greet.')
    click.option('--min_flexible_attributes', prompt='Min flexible attributes', help='The person to greet.')
    click.option('--min_undesired_support', prompt='Min undesired support', help='The person to greet.')
    click.option('--min_undesired_confidence', prompt='Min undesired confidence', help='The person to greet.')
    click.option('--min_desired_support', prompt='Min desired support', help='The person to greet.')
    click.option('--min_desired_confidence', prompt='Min desired confidence', help='The person to greet.')
    click.option('--csv_path', prompt='CSV Path', help='The person to greet.')
    click.option('--stable_attributes', prompt='Stable attributes', help='The person to greet.')
    click.option('--flexible_attributes', prompt='Flexible attributes', help='The person to greet.')
    click.option('--target', prompt='Target', help='The person to greet.')
    click.option('--undesired_state', prompt='Undesired state', help='The person to greet.')
    click.option('--desired_state', prompt='Desired state', help='The person to greet.')
    click.option('--output_json_path', prompt='Output CSV Path', help='The person to greet.')


if __name__ == "__main__":
    main()  # pragma: no cover
