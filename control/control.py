from desiredstate import DesiredState
from decisions import Decisions
from reduction import Reduction
from arules import ActionRules


class Control:
    """
    Control class
    """

    def __init__(self):
        self.decisions = Decisions()
        self.action_rules = ()

    def read_csv(self, file, **kwargs):
        # Find all couples of classification rules and try to create action rules
        self.decisions.read_csv(file, **kwargs)

    def load_pandas(self, dataframe):
        self.decisions.load_pandas(dataframe)

    def fit(self, desired_classes, stable_antecedents, flexible_antecedents, consequent, conf, supp, is_nan=False):
        desired_state = DesiredState(desired_classes)
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
        arules = ActionRules(
            reduced_tables.stable_tables,
            reduced_tables.flexible_tables,
            reduced_tables.decision_tables,
            desired_state,
            reduced_tables.supp,
            reduced_tables.conf,
            is_nan
        )
        arules.fit()
        self.action_rules = arules.action_rules

    def get_action_rules(self):
        return self.action_rules
