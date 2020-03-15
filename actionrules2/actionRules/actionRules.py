import pandas as pd
from typing import List
from typing import Union
import itertools

from actionrules.desiredState import DesiredState


class ActionRules:
    """
    The class ActionRules is the one where the algorithm for action rules discovery is settled.

    ...

    Attributes
    ----------
    stable_tables : List[pd.DataFrame]
        Data frames with stable attributes.
    flexible_tables : List[pd.DataFrame]
        Data frames with flexible attributes.
    decision_tables : List[pd.DataFrame]
        Data frames with consequent.
    desired_state : DesiredState()
        DesiredState object.
    action_rules : list
        Discovered action rules.
    action_rules_pretty_text : list
        Readable discovered action rules.
    action_rules_representation : list
        Math representation of action rules.
    supp : List[pd.Series]
        List od supports for classification rules.
    conf : List[pd.Series]
        List od confidences for classification rules.
    is_nan : bool
        True means NaN values are used, False means NaN values are not used.
    min_stable_antecedents : int
        Minimal number of stable pairs.
    min_flexible_antecedents : int
        Minimal number of flexible pairs.
    max_stable_antecedents : int
        Maximal number of stable pairs.
    max_flexible_antecedents : int
        Maximal number of flexible pairs.
    used_indexes : list
        Already used indexes.
    classification_before : list
        List of before parts of action rules.
    classification_after : list
        List of after parts of action rules.

    Methods
    -------
    fit(self)
        Train the model.
    pretty_text(self)
        Generate pretty representation of action rules.
    representation(self)
        Generate mathematical representation of action rules.

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
                 max_flexible_antecedents: int = 1,
                 ):
        """
        Parameters
        ----------
        stable_tables : List[pd.DataFrame]
            Data frames with stable attributes.
        flexible_tables : List[pd.DataFrame]
            Data frames with flexible attributes.
        decision_tables : List[pd.DataFrame]
            Data frames with consequent.
        desired_state : DesiredState()
            DesiredState object.
        supp : List[pd.Series]
            List od supports for classification rules.
        conf : List[pd.Series]
            List od confidences for classification rules.
        is_nan : bool
            True means NaN values are used, False means NaN values are not used.
        min_stable_antecedents : int
            Minimal number of stable pairs.
        min_flexible_antecedents : int
            Minimal number of flexible pairs.
        max_stable_antecedents : int
            Maximal number of stable pairs.
        max_flexible_antecedents : int
            Maximal number of flexible pairs.
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
        self.classification_before = []
        self.classification_after = []

    def _is_action_couple(self,
                          before: Union[str, int, float],
                          after: Union[str, int, float],
                          attribute_type: str
                          ) -> tuple:
        """ Check if the state before and after can make action rule.

        Parameters
        ----------
        before : Union[str, int, float]
            Before part of the candidate.
        after : Union[str, int, float]
            After part of the candidate.
        attribute_type : str
            Attribute type (stable or flexible).

        Returns
        -------
        tuple
            Returns (bool is_action_pair, (before, after) action_pair, bool break_rule).
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
                elif before != "nan" and after == "nan":
                    return False, None, False
        elif attribute_type == "flexible":
            if before == "nan" and after == "nan":
                return False, None, False
            elif before != after and before != "nan" and after != "nan":
                return True, (before, after), False
            elif self.is_nan:
                if before != after and before == "nan":
                    return True, (str(None), after), False
                if before != after and after == "nan":
                    return False, None, False
        return False, None, True

    def _create_action_rules(self,
                             df: pd.DataFrame,
                             rule_before_index: int,
                             rule_after_index: int,
                             attribute_type: str) -> tuple:
        """It creates action rules pairs.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame with classification rules.
        rule_before_index : int
            Candidate before index.
        rule_after_index : int
            Candidate after index.
        attribute_type : str
            Type of attributes in the data frame (stable or flexible)

        Returns
        -------
        bool
            Does it break the condition to be an action rule?
        list
            Generated part of an action rule.
        int
            Number of used attributes in antecedent.
        """
        action_rule_part = []
        count_antecedent = 0
        columns = list(df)
        for column in columns:
            is_action_couple, action_couple, break_rule = self._is_action_couple(
                before=df[column][rule_before_index],
                after=df[column][rule_after_index],
                attribute_type=attribute_type)
            if break_rule:
                return False, None, None
            elif is_action_couple:
                count_antecedent += 1
                action_rule_part.append([column, action_couple])
            else:
                if action_couple is not None:
                    action_rule_part.append([column, action_couple])
        return True, action_rule_part, count_antecedent

    def _add_action_rule(self,
                         action_rule_stable: list,
                         action_rule_flexible: list,
                         action_rule_decision: list,
                         action_rule_supp: list,
                         action_rule_conf: list):
        """This method joins the parts of an action rule and adds the action rule to a list.

        Parameters
        ----------
        action_rule_stable : list
            List of stable attributes.
        action_rule_flexible : list
            List of actions in flexible attributes.
        action_rule_decision : list
            List of changes in consequent.
        action_rule_supp : list
            List of supports.
        action_rule_conf : list
            List of confidences.
        """
        action_rule = [action_rule_stable, action_rule_flexible, action_rule_decision]
        uplift = self._get_uplift(action_rule_supp[0], action_rule_conf[0], action_rule_conf[1])
        self.action_rules.append([action_rule, action_rule_supp, action_rule_conf, uplift])

    def fit(self):
        """It finds all pairs of classification rules and tries to create action rules.

        """
        for table in range(len(self.stable_tables)):
            stable_columns = self.stable_tables.pop(0)
            flexible_columns = self.flexible_tables.pop(0)
            decision_column = self.decision_tables.pop(0)
            supp = self.supp.pop(0)
            conf = self.conf.pop(0)
            indexes = list(stable_columns.index.values)
            for comb in itertools.permutations(indexes, 2):
                # Check if it is not used twice - just for reduction by nan
                if self.is_nan:
                    if comb in self.used_indexes:
                        continue
                    self.used_indexes.append(comb)
                rule_before_index = comb[0]
                rule_after_index = comb[1]
                decision_before = decision_column.at[rule_before_index, decision_column.columns[0]]
                decision_after = decision_column.at[rule_after_index, decision_column.columns[0]]
                if self.desired_state.is_candidate_decision(decision_before, decision_after):
                    is_all_stable, action_rule_stable, counted_stable = self._create_action_rules(
                        stable_columns,
                        rule_before_index,
                        rule_after_index,
                        "stable")
                    if not is_all_stable:
                        continue
                    is_all_flexible, action_rule_flexible, counted_flexible = self._create_action_rules(
                        flexible_columns,
                        rule_before_index,
                        rule_after_index,
                        "flexible")
                    if not is_all_flexible:
                        continue
                    action_rule_decision = [
                        decision_column.columns[0], [decision_before, decision_after]]
                    if counted_flexible >= self.min_flexible_antecedents and \
                       counted_stable >= self.min_stable_antecedents and \
                       counted_flexible <= self.max_flexible_antecedents and \
                       counted_stable <= self.max_stable_antecedents:
                        action_rule_supp = [supp[rule_before_index],
                                            supp[rule_after_index],
                                            min(supp[rule_before_index], supp[rule_after_index])
                                            ]
                        action_rule_conf = [conf[rule_before_index],
                                            conf[rule_after_index],
                                            conf[rule_before_index] * conf[rule_after_index]
                                            ]
                        self._add_action_rule(action_rule_stable,
                                              action_rule_flexible,
                                              action_rule_decision,
                                              action_rule_supp,
                                              action_rule_conf)
                        self.classification_before.append(rule_before_index)
                        self.classification_after.append(rule_after_index)

    def pretty_text(self):
        """It generates human language representation of action rules.

        """
        for row in self.action_rules:
            action_rule = row[0]
            supp = row[1]
            conf = row[2]
            uplift = row[3]
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
                    str(decision[1][1]) + "' with support: " + str(supp[2]) + ", confidence: " + str(conf[2]) + \
            " and uplift: " + str(uplift) + "."
            self.action_rules_pretty_text.append(text)

    def representation(self):
        """It generates a mathematical representation of action rules.

        """
        for row in self.action_rules:
            action_rule = row[0]
            supp = row[1]
            conf = row[2]
            uplift = row[3]
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
                    str(decision[1][1]) + "] with support: " + str(supp[2]) + ", confidence: " + str(
                conf[2]) + " and uplift: " + str(uplift) + "."
            self.action_rules_representation.append(text)

    @staticmethod
    def _get_uplift(supp_before: float, conf_before: float, conf_after: float) -> float:
        """Get uplift for action rule.

        Uplift = P(target|treatment) -  P(target|no treatment)

        Parameters
        ----------
        supp_before: float
            Support before.
        conf_before: float
            Confidence before.
        conf_after: float
            Confidence after.

        Returns
        -------
        float
            An uplift value.
        """
        return ((supp_before / conf_before) * conf_after) - ((supp_before / conf_before) - supp_before)
