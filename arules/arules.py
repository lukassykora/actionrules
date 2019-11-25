import pandas as pd
from typing import List
from typing import Union
import itertools

from ..desiredstate import DesiredState


class ActionRules:
    """
    Check all classification couples if they can make action rule
    """

    def __init__(self,
                 stable_tables: List[pd.DataFrame],
                 flexible_tables: List[pd.DataFrame],
                 decision_tables: List[pd.DataFrame],
                 desired_state: DesiredState,
                 supp: List[pd.Series],
                 conf: List[pd.Series],
                 is_nan: bool = False,
                 min_stable_antecedents: int = 1,
                 min_flexible_antecedents: int = 1,
                 max_stable_antecedents: int = 1,
                 max_flexible_antecedents: int = 1
                 ):
        """
        Initialise by reduced tables.
        """
        self.stable_tables = stable_tables
        self.flexible_tables = flexible_tables
        self.decision_tables = decision_tables
        self.desired_state = desired_state
        self.action_rules = []
        self.action_rules_pretty_text = []
        self.action_rules_representation = []
        self.supp = supp
        self.conf = conf
        self.is_nan = is_nan
        self.min_stable_antecedents = min_stable_antecedents
        self.min_flexible_antecedents = min_flexible_antecedents
        self.max_stable_antecedents = max_stable_antecedents
        self.max_flexible_antecedents = max_flexible_antecedents
        self.used_indexes = []

    def _is_action_couple(self,
                          before: Union[str, int, float],
                          after: Union[str, int, float],
                          attribute_type: str
                          ) -> tuple:
        """
        Check if the state before and after make action rule.
        Return acton rule part and if the supp and conf can be used.
        """
        before = str(before)
        after = str(after)
        if attribute_type == "stable":
            if before == "nan" and after == "nan":
                return False, None, False
            elif before == after and before != "nan":
                return True, (before,), False
            elif self.is_nan:
                if before == "nan" and after != "nan":
                    return True, (after + "*",), False
        elif attribute_type == "flexible":
            if before == "nan" and after == "nan":
                return False, None, False
            elif before != after and before != "nan" and after != "nan":
                return True, (before, after), False
            elif self.is_nan:
                if before != after and before == "nan":
                    return True, (str(None), after), False
        return False, None, True

    def _create_action_rules(self, couple: pd.DataFrame, attribute_type: str) -> tuple:
        """
        Create action rules couples
        """
        action_rule_part = []
        count_antecedents = 0
        for column in couple:
            is_action_couple, action_couple, break_rule = self._is_action_couple(
                before=couple.at[0, column],
                after=couple.at[1, column],
                attribute_type=attribute_type)
            if break_rule:
                return False, None, None
            elif is_action_couple:
                count_antecedents += 1
                action_rule_part.append([column, action_couple])
            else:
                if action_couple is not None:
                    action_rule_part.append([column, action_couple])
        return True, action_rule_part, count_antecedents

    def _add_action_rule(self,
                         action_rule_stable: list,
                         action_rule_flexible: list,
                         action_rule_decision: list,
                         action_rule_supp: list,
                         action_rule_conf: list):
        """
        Add action rule to list
        """
        action_rule = [action_rule_stable, action_rule_flexible, action_rule_decision]
        self.action_rules.append([action_rule, action_rule_supp, action_rule_conf])

    def fit(self):
        """
        Find all couples of classification rules and try to create action rules
        """
        for table in range(len(self.stable_tables)):
            stable_columns = self.stable_tables.pop(0)
            flexible_columns = self.flexible_tables.pop(0)
            decision_column = self.decision_tables.pop(0)
            supp = self.supp.pop(0)
            conf = self.conf.pop(0)
            indexes = list(stable_columns.index.values)
            for comb in itertools.permutations(indexes, 2):
                if comb in self.used_indexes:
                    continue
                self.used_indexes.append(comb)
                stable_couples = stable_columns.loc[list(comb)].reset_index(drop=True)
                flexible_couples = flexible_columns.loc[list(comb)].reset_index(drop=True)
                decision_couples = decision_column.loc[list(comb)].reset_index(drop=True)
                supp_couples = supp.loc[list(comb)].reset_index(drop=True)
                conf_couples = conf.loc[list(comb)].reset_index(drop=True)
                if self.desired_state.is_candidate_couple(decision_couples):
                    is_all_stable, action_rule_stable, counted_stable = self._create_action_rules(
                        stable_couples,
                        "stable")
                    is_all_flexible, action_rule_flexible, counted_flexible = self._create_action_rules(
                        flexible_couples,
                        "flexible")
                    action_rule_decision = [
                        decision_couples.columns[0], [decision_couples.iat[0, 0], decision_couples.iat[1, 0]]]
                    if is_all_stable and \
                            is_all_flexible and \
                            counted_flexible >= self.min_flexible_antecedents and \
                            counted_stable >= self.min_stable_antecedents and \
                            counted_flexible <= self.max_flexible_antecedents and \
                            counted_stable <= self.max_stable_antecedents:
                        action_rule_supp = [supp_couples[0], supp_couples[1], min(supp_couples[0], supp_couples[1])]
                        action_rule_conf = [conf_couples[0], conf_couples[1], conf_couples[0] * conf_couples[1]]
                        self._add_action_rule(action_rule_stable,
                                              action_rule_flexible,
                                              action_rule_decision,
                                              action_rule_supp,
                                              action_rule_conf)

    def pretty_text(self):
        for row in self.action_rules:
            action_rule = row[0]
            supp = row[1]
            conf = row[2]
            text = "If "
            # Stable part
            stable_part = action_rule[0]
            for stable_couple in stable_part:
                text += "attribute '" + str(stable_couple[0]) + "' is '" + str(stable_couple[1][0]) + "', "
            # Flexible part
            flexible_part = action_rule[1]
            for flexible_couple in flexible_part:
                text += "attribute '" + str(flexible_couple[0]) + "' value '" + str(flexible_couple[1][0]) + \
                        "' is changed to '" + str(flexible_couple[1][1]) + "', "
            # Decision
            decision = action_rule[2]
            text += "then '" + str(decision[0]) + "' value '" + str(decision[1][0]) + "' is changed to '" + \
                    str(decision[1][1]) + "' with support: " + str(supp[2]) + " and confidence: " + str(conf[2]) + "."
            self.action_rules_pretty_text.append(text)

    def representation(self):
        for row in self.action_rules:
            action_rule = row[0]
            supp = row[1]
            conf = row[2]
            text = "r = [   "
            # Stable part
            stable_part = action_rule[0]
            text = text[:-3]
            for stable_couple in stable_part:
                text += "(" + str(stable_couple[0]) + ": " + str(stable_couple[1][0]) + ") ∧ "
            # Flexible part
            flexible_part = action_rule[1]
            text = text[:-3]
            for flexible_couple in flexible_part:
                text += " ∧ (" + str(flexible_couple[0]) + ": " + str(flexible_couple[1][0]) + \
                        " → " + str(flexible_couple[1][1]) + ") "
            # Decision
            decision = action_rule[2]
            text += "] ⇒ [" + str(decision[0]) + ": " + str(decision[1][0]) + " → " + \
                    str(decision[1][1]) + "] with support: " + str(supp[2]) + " and confidence: " + str(
                conf[2])
            self.action_rules_representation.append(text)
