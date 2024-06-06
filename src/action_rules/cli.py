"""Console script for action_rules."""

import os
from typing import TextIO

import click
import pandas as pd

from action_rules import ActionRules

NUMBER_OF_COLUMNS_DISPLAYED = 20


@click.command()
@click.option(
    '--min_stable_attributes',
    prompt='Min. stable attributes',
    help='Minimum number of stable attributes.',
    default=1,
)
@click.option(
    '--min_flexible_attributes',
    prompt='Min. flexible attributes',
    help='Minimum number of flexible attributes. Must be at least 1.',
    default=1,
)
@click.option(
    '--min_undesired_support',
    prompt='Min. undesired support',
    help='Support of the undesired part of the rule. Number of instances matching all conditions in the '
    'antecedent and consequent of r1.',
)
@click.option(
    '--min_undesired_confidence',
    prompt='Min. undesired confidence',
    help='Confidence of the undesired part of the rule. Number of instances matching all conditions in the '
    'antecedent and consequent of r1 divided by number of instances matching all conditions in the '
    'antecedent of r1.',
)
@click.option(
    '--min_desired_support',
    prompt='Min. desired support',
    help='Support of the desired part of the rule. Number of instances matching all conditions in the antecedent '
    'and consequent of r2.',
)
@click.option(
    '--min_desired_confidence',
    prompt='Min. desired confidence',
    help='Confidence of the desired part of the rule. Number of instances matching all conditions in the '
    'antecedent and consequent of r2 divided by number of instances matching all conditions in the '
    'antecedent of r2.',
)
@click.option(
    '--csv_path',
    type=click.File('rb'),
    prompt='CSV Path',
    help='Dataset where the first row is the header. A comma is used as a separator.',
)
@click.option(
    '--stable_attributes',
    prompt='Stable attributes (comma separated)',
    help='These attributes remain unchanged regardless of the actions described by the rule.',
)
@click.option(
    '--flexible_attributes',
    prompt='Flexible attributes (comma separated)',
    help='These are the attributes that can change.',
)
@click.option('--target', prompt='Target', help='This is the outcome attribute that the action rule aims to influence.')
@click.option(
    '--undesired_state',
    prompt='Undesired state',
    help='The undesired state of a target is the current or starting state that you want to change or improve. It '
    'represents a negative or less preferred outcome.',
)
@click.option(
    '--desired_state',
    prompt='Desired state',
    help='The desired state of a target is goal state that you want to achieve as a result of applying the action '
    'rule. It represents a positive or preferred outcome.',
)
@click.option(
    '--output_json_path',
    type=click.File('wb'),
    prompt='Output CSV Path',
    help='Action Rules (JSON representation).',
    default='rules.json',
)
def main(
    min_stable_attributes: int,
    min_flexible_attributes: int,
    min_undesired_support: int,
    min_undesired_confidence: float,
    min_desired_support: int,
    min_desired_confidence: float,
    csv_path: TextIO,
    stable_attributes: str,
    flexible_attributes: str,
    target: str,
    undesired_state: str,
    desired_state: str,
    output_json_path: TextIO,
):
    """Entrypoint for console script."""
    click.echo("action-rules")
    click.echo("=" * len("action-rules"))
    click.echo("The package for action rules mining using Action-Apriori (Apriori Modified for Action Rules Mining).")
    click.echo(
        "A single action rule can actually be decomposed to two rules r1 and r2, one representing the state before ("
        "undesired part) and the second one after the intervention (desired part): r1 -> r2."
    )
    action_rules = ActionRules(
        int(min_stable_attributes),
        int(min_flexible_attributes),
        int(min_undesired_support),
        float(min_undesired_confidence),
        int(min_desired_support),
        float(min_desired_confidence),
    )
    data = pd.read_csv(os.path.abspath(csv_path.name))
    cols = list(data.columns)
    if len(cols) < NUMBER_OF_COLUMNS_DISPLAYED:
        cols_string = ', '.join(cols) + '.'
    else:
        cols_string = ', '.join(cols[:NUMBER_OF_COLUMNS_DISPLAYED]) + ', ...'
    click.echo("Columns: " + cols_string)
    action_rules.fit(
        data,
        [x.strip() for x in str(stable_attributes).split(",")],
        [x.strip() for x in str(flexible_attributes).split(",")],
        str(target),
        str(undesired_state),
        str(desired_state),
    )
    rules = action_rules.get_rules()
    click.echo(rules.action_rules)
    if rules is not None:
        output_json_path.write(str.encode(str(rules.get_export_notation())))
    click.echo("Done.")


if __name__ == "__main__":
    main()  # pragma: no cover
