"""Console script for action_rules."""

import os
from typing import BinaryIO

import click
import pandas as pd

from action_rules import ActionRules


@click.command()
@click.option(
    '--min_stable_attributes',
    prompt='Min. stable attributes',
    help='Minimum number of stable attributes.',
    default=1,
    type=int,
)
@click.option(
    '--min_flexible_attributes',
    prompt='Min. flexible attributes',
    help='Minimum number of flexible attributes. Must be at least 1.',
    default=1,
    type=int,
)
@click.option(
    '--min_undesired_support',
    prompt='Min. undesired support',
    help='Support of the undesired part of the rule. Number of instances matching all conditions in the '
    'antecedent and consequent of r1.',
    type=int,
)
@click.option(
    '--min_undesired_confidence',
    prompt='Min. undesired confidence',
    help='Confidence of the undesired part of the rule. Number of instances matching all conditions in the '
    'antecedent and consequent of r1 divided by number of instances matching all conditions in the '
    'antecedent of r1.',
    type=float,
)
@click.option(
    '--min_desired_support',
    prompt='Min. desired support',
    help='Support of the desired part of the rule. Number of instances matching all conditions in the antecedent '
    'and consequent of r2.',
    type=int,
)
@click.option(
    '--min_desired_confidence',
    prompt='Min. desired confidence',
    help='Confidence of the desired part of the rule. Number of instances matching all conditions in the '
    'antecedent and consequent of r2 divided by number of instances matching all conditions in the '
    'antecedent of r2.',
    type=float,
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
    type=str,
)
@click.option(
    '--flexible_attributes',
    prompt='Flexible attributes (comma separated)',
    help='These are the attributes that can change.',
    type=str,
)
@click.option('--target', prompt='Target', help='This is the outcome attribute that the action rule aims to influence.')
@click.option(
    '--undesired_state',
    prompt='Undesired state',
    help='The undesired state of a target is the current or starting state that you want to change or improve. It '
    'represents a negative or less preferred outcome.',
    type=str,
)
@click.option(
    '--desired_state',
    prompt='Desired state',
    help='The desired state of a target is goal state that you want to achieve as a result of applying the action '
    'rule. It represents a positive or preferred outcome.',
    type=str,
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
    csv_path: BinaryIO,
    stable_attributes: str,
    flexible_attributes: str,
    target: str,
    undesired_state: str,
    desired_state: str,
    output_json_path: BinaryIO,
):
    """
    CLI.

    Decompose a single action rule into two rules, r1 and r2: one representing the state before (undesired part) and
    the other after the intervention (desired part): r1 -> r2.

    Parameters
    ----------
    min_stable_attributes : int
        Minimum number of stable attributes required.
    min_flexible_attributes : int
        Minimum number of flexible attributes required.
    min_undesired_support : int
        Minimum support for the undesired state.
    min_undesired_confidence : float
        Minimum confidence for the undesired state.
    min_desired_support : int
        Minimum support for the desired state.
    min_desired_confidence : float
        Minimum confidence for the desired state.
    csv_path : BinaryIO
        Path to the CSV file containing the dataset.
    stable_attributes : str
        Comma-separated list of stable attributes.
    flexible_attributes : str
        Comma-separated list of flexible attributes.
    target : str
        Target attribute for the action rule.
    undesired_state : str
        The undesired state of the target attribute.
    desired_state : str
        The desired state of the target attribute.
    output_json_path : BinaryIO
        Path to the output JSON file where the results will be saved.

    Returns
    -------
    None
    """
    click.echo("action-rules")
    click.echo("=" * len("action-rules"))
    click.echo("The package for action rules mining using Action-Apriori (Apriori Modified for Action Rules Mining).")
    action_rules = ActionRules(
        int(min_stable_attributes),
        int(min_flexible_attributes),
        int(min_undesired_support),
        float(min_undesired_confidence),
        int(min_desired_support),
        float(min_desired_confidence),
    )
    data = pd.read_csv(os.path.abspath(csv_path.name))
    action_rules.fit(
        data,
        [x.strip() for x in str(stable_attributes).split(",")],
        [x.strip() for x in str(flexible_attributes).split(",")],
        str(target),
        str(undesired_state),
        str(desired_state),
    )
    rules = action_rules.get_rules()
    if rules is not None:
        output_json_path.write(str.encode(str(rules.get_export_notation())))
    click.echo("Done.")


if __name__ == "__main__":
    main()  # pragma: no cover
