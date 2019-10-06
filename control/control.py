from typing import List
from typing import Tuple
from desiredstate import DesiredState
from decisions import Decisions
from reduction import Reduction
from arules import ActionRules
import pandas as pd


class Control:
    """
    Control class
    """

    def __init__(self):
        """
        Initialise
        """
        self.decisions = Decisions()
        self.action_rules = ()

    def read_csv(self, file: str, **kwargs):
        """
        Create data frame from csv
        """
        # Find all couples of classification rules and try to create action rules
        self.decisions.read_csv(file, **kwargs)

    def load_pandas(self, data_frame: pd.DataFrame):
        """
        Load data frame
        """
        self.decisions.load_pandas(data_frame)

    def fit(self,
            stable_antecedents: List[str],
            flexible_antecedents: List[str],
            consequent: str,
            conf: float,
            supp: float,
            desired_classes: List[str] = None,
            desired_changes: List[list] = None,
            is_nan: bool=False):
        """
        Get action rules.
        Define antecedent and consequent.
        - stable_antecedents - List of column names.
        - flexible_antecedents - List of column names.
        - consequent - Name of column.
        Confidence and support.
        - conf - value in % for confidence of classification rules.
        - supp - value in % for support of classification rules.
        Desired classes or desired changes must be entered.
        - desired_classes - List of decision states. For example ["1"].
        - desired_changes - List of desired changes. For example [["0", "1"]].
        Should nan values be used.
        - is_nan - True means nan values are used, False means nan values are not used.
        """
        if bool(desired_classes) != bool(desired_changes):
            desired_state = DesiredState(desired_classes=desired_classes, desired_changes=desired_changes)
        else:
            raise Exception("Desired classes or desired changes must be entered")
        antecedents = stable_antecedents + flexible_antecedents
        self.decisions.prepare_data_fim(antecedents, consequent)
        self.decisions.fit_fim_apriori(conf=conf, support=supp)
        self.decisions.generate_decision_table()
        stable = self.decisions.decision_table[stable_antecedents]
        flex = self.decisions.decision_table[flexible_antecedents]
        target = self.decisions.decision_table[[consequent]]
        supp = self.decisions.support
        conf = self.decisions.confidence
        reduced_tables = Reduction(stable, flex, target, desired_state, supp, conf)
        reduced_tables.reduce()
        action_rules = ActionRules(
            reduced_tables.stable_tables,
            reduced_tables.flexible_tables,
            reduced_tables.decision_tables,
            desired_state,
            reduced_tables.supp,
            reduced_tables.conf,
            is_nan
        )
        action_rules.fit()
        self.action_rules = action_rules.action_rules

    def get_action_rules(self) -> Tuple:
        """
        Get action rules.
        """
        return self.action_rules
