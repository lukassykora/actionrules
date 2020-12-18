from typing import List
import pandas as pd

from actionrules.desiredState import DesiredState
from actionrules.decisions import Decisions
from actionrules.reduction import Reduction
from actionrules.actionRules import ActionRules


class ActionRulesDiscovery:
    """
    The class ActionRulesDiscovery is the main control class where the methods fit or
    predict can be called. However, there is a minimum logic in the class. It is just an
    interface that initializes other objects and calls their methods.

    ...

    Attributes
    ----------
    decisions : Decisions()
        Object that runs PyFIM classification rules discovery.
    action_rules: None or ActionRules
        Object that runs action rules discovery.
    desired_state: DesiredState
        Object that is responsible for handling of desired state.
    stable_attributes: List[str]
        List of stable attributes.
    flexible_attributes: List[str]
        List of flexible attributes.
    consequent: str
        Name of consequent columns.

    Methods
    -------
    read_csv(self, file: str, **kwargs)
        Import data from a CSV file.
    load_pandas(self, data_frame: pd.DataFrame)
        Import data from Pandas data frame.
    fit(self,
        stable_attributes: List[str],
        flexible_attributes: List[str],
        consequent: str,
        conf: float,
        supp: float,
        desired_classes: List[str] = None,
        desired_changes: List[list] = None,
        is_nan: bool = False,
        is_reduction: bool = True,
        min_stable_attributes: int = 1,
        min_flexible_attributes: int = 1,
        max_stable_attributes: int = 5,
        max_flexible_attributes: int = 5,
        )
        Train the model from transaction data.
    fit_classification_rules(self,
                             stable_attributes: List[str],
                             flexible_attributes: List[str],
                             consequent: str,
                             conf_col: str,
                             supp_col: str,
                             desired_classes: List[str] = None,
                             desired_changes: List[list] = None,
                             is_nan: bool = False,
                             is_reduction: bool = True,
                             min_stable_attributes: int = 1,
                             min_flexible_attributes: int = 1,
                             max_stable_attributes: int = 5,
                             max_flexible_attributes: int = 5,
                             )
        Train the model from classification rules.
    get_action_rules(self) -> list
        Get list of action rules (machine representation)
    get_pretty_action_rules(self) -> list
        Get human-readable representations of action rules
    get_action_rules_representation(self) -> list
        Get math representation of action rules
    get_source_data_for_ar(self, action_r_number: int, is_before: bool) -> pd.DataFrame
        Get the source data the action rule is discovered from.
    predict(self, source_table: pd.DataFrame) -> pd.DataFrame
        Predicts if new occurrence would need any change.

    """
    ACTION_RULE = "action rule"
    ACTION_RULE_TARGET = "action rule target"
    SUPPORT_BEFORE = "support before"
    SUPPORT_AFTER = "support after"
    ACTION_RULE_SUPPORT = "action rule support"
    CONFIDENCE_BEFORE = "confidence before"
    CONFIDENCE_AFTER = "confidence after"
    ACTION_RULE_CONFIDENCE = "action rule confidence"
    ACTION_RULE_UPLIFT = "uplift"
    RECOMMENDED = "-recommended"

    def __init__(self):
        """
        Initialise
        """
        self.decisions = Decisions()
        self.action_rules = None
        self.desired_state = None
        self.stable_attributes = []
        self.flexible_attributes = []
        self.consequent = ""

    def _check_columns(self, attributes: List[str], consequent: str):
        """Checks if inserted data is valid (columns exist, rows exist).

        Parameters
        ----------
        attributes : List[str]
            List of attributes names.
        consequent : str
            The name of consequent.

        Raises
        ------
        Exception
            If no data entered or column does not exist in data.
        """
        if len(self.decisions.data) == 0:
            raise Exception("No data entered.")
        columns = self.decisions.data.columns
        for col_name in attributes + [consequent]:
            if col_name not in columns:
                raise Exception("Column " + str(col_name) + " does not exist in data")

    def read_csv(self, file: str, **kwargs):
        """Imports data from a CSV file.

        It uses the same optional parameters as read_csv from Pandas.

        Parameters
        ----------
        file : str
            A path to a file.
        **kwargs :
            Arbitrary keyword arguments (the same as in Pandas).
        """
        self.decisions.read_csv(file, **kwargs)

    def load_pandas(self, data_frame: pd.DataFrame):
        """Loads a data frame.

        It must be the Pandas data frame.

        Parameters
        ----------
        data_frame : pd.DataFrame
            Pandas data frame.
        """
        self.decisions.load_pandas(data_frame)

    def fit(self,
            stable_attributes: List[str],
            flexible_attributes: List[str],
            consequent: str,
            conf: float,
            supp: float,
            desired_classes: List[str] = None,
            desired_changes: List[list] = None,
            is_nan: bool = False,
            is_reduction: bool = True,
            min_stable_attributes: int = 1,
            min_flexible_attributes: int = 1,
            max_stable_attributes: int = 5,
            max_flexible_attributes: int = 5,
            is_strict_flexible: bool = True
            ):
        """Train the model from transaction data.

        Define antecedent and consequent.
        - stable_attributes
        - flexible_attributes
        - consequent
        Confidence and support.
        - conf
        - supp
        Desired classes or desired changes must be entered.
        - desired_classes
        - desired_changes
        Should uncertainty be used?
        - is_nan
        Should the reduction table be used?
        - is_reduction
        Minimal number of stable and flexible pairs in antecedent.
        - min_stable_attributes
        - min_flexible_attributes
        - max_stable_attributes
        - max_flexible_attributes
        The way how the flexible attribute behave
        - is_strict_flexible

        Parameters
        ----------
        stable_attributes : List[str]
            List of column names.
        flexible_attributes : List[str]
            List of column names.
        consequent : str
            Name of the consequent column.
        conf : float
            Value in % for minimal confidence in classification rules.
            For example, 60.
        supp : float
            Value in % for minimal support of classification rules.
            For example, 5.
        desired_classes : List[str] = None
            List of decision states. For example, ["1"].
            DEFAULT: None
        desired_changes : List[list] = None
            List of desired changes. For example, [["0", "1"]].
            DEFAULT: None
        is_nan : bool = False
            True means NaN values are used, False means NaN values are not used.
            It means NaN values from classification rules.
            If is_nan is true, the uncertainty is used.
            DEFAULT: FALSE
        is_reduction : bool = True
            Is the reduction table used?
            DEFAULT: TRUE
        min_stable_attributes : int = 1
            Minimal number of stable pairs.
            DEFAULT: 1
        min_flexible_attributes : int = 1
            Minimal number of flexible pairs.
            DEFAULT: 1
        max_stable_attributes : int = 5
            Maximal number of stable pairs.
            DEFAULT: 5
        max_flexible_attributes : int = 5
            Maximal number of flexible pairs.
            DEFAULT: 5
        is_strict_flexible : bool = True
            If true flexible attributes must be always actionable, if false they can also behave as stable attributes
            DEFAULT: True
        """
        if (self.action_rules):
            raise Exception("Fit was already called")
        self.consequent = consequent
        if bool(desired_classes) != bool(desired_changes):
            self.desired_state = DesiredState(desired_classes=desired_classes, desired_changes=desired_changes)
        else:
            raise Exception("Desired classes or desired changes must be entered")
        attributes = stable_attributes + flexible_attributes
        self._check_columns(attributes, consequent)
        self.decisions.prepare_data_fim(attributes, consequent)
        self.decisions.fit_fim_apriori(conf=conf, support=supp)
        self.decisions.generate_decision_table()
        # Not all columns are in the generated classification rules
        self.stable_attributes = list(set(stable_attributes).intersection(set(self.decisions.decision_table.columns)))
        self.flexible_attributes = list(set(flexible_attributes).intersection(set(self.decisions.decision_table.columns)))
        # Data
        stable = self.decisions.decision_table[self.stable_attributes]
        flex = self.decisions.decision_table[self.flexible_attributes]
        target = self.decisions.decision_table[[consequent]]
        supp = self.decisions.support
        conf = self.decisions.confidence
        reduced_tables = Reduction(stable, flex, target, self.desired_state, supp, conf, is_nan)
        if is_reduction:
            reduced_tables.reduce()
        self.action_rules = ActionRules(
            reduced_tables.stable_tables,
            reduced_tables.flexible_tables,
            reduced_tables.decision_tables,
            self.desired_state,
            self.decisions,
            reduced_tables.supp,
            reduced_tables.conf,
            is_nan,
            min_stable_attributes,
            min_flexible_attributes,
            max_stable_attributes,
            max_flexible_attributes,
            is_strict_flexible
        )
        self.action_rules.fit()

    def fit_classification_rules(self,
                                 stable_attributes: List[str],
                                 flexible_attributes: List[str],
                                 consequent: str,
                                 conf_col: str,
                                 supp_col: str,
                                 desired_classes: List[str] = None,
                                 desired_changes: List[list] = None,
                                 is_nan: bool = False,
                                 is_reduction: bool = True,
                                 min_stable_attributes: int = 1,
                                 min_flexible_attributes: int = 1,
                                 max_stable_attributes: int = 5,
                                 max_flexible_attributes: int = 5,
                                 is_strict_flexible: bool = True
                                 ):
        """Train the model from classification rules.

        Define antecedent and consequent.
        - stable_attributes
        - flexible_attributes
        - consequent
        Confidence and support.
        - conf_col
        - supp_col
        Desired classes or desired changes must be entered.
        - desired_classes
        - desired_changes
        Should uncertainty be used?
        - is_nan
        Should the reduction table be used?
        - is_reduction
        Minimal number of stable and flexible pairs in antecedent.
        - min_stable_attributes
        - min_flexible_attributes
        - max_stable_attributes
        - max_flexible_attributes
        The way how the flexible attribute behave
        - is_strict_flexible

        Parameters
        ----------
        stable_attributes: List[str]
            List of column names.
        flexible_attributes: List[str]
            List of column names.
        consequent: str
            Name of the consequent column.
        conf_col: str
            Name of the column with classification rule confidence -
            the numbers should be in form 0.1 for 10%.
        supp_col: str
            Name of the column with classification rule support -
            the numbers should be in form 0.1 for 10%.
        desired_classes: List[str] = None
            List of decision states. For example ["1"].
            DEFAULT: None
        desired_changes: List[list] = None
            List of desired changes. For example [["0", "1"]].
            DEFAULT: None
        is_nan: bool = False
            True means NaN values are used, False means nan values are not used.
            If is_nan is true, the uncertainty is used.
            DEFAULT: FALSE
        is_reduction: bool = True
            Is the reduction table used?
            DEFAULT: TRUE
        min_stable_attributes: int = 1
            Minimal number of stable pairs.
            DEFAULT: 1
        min_flexible_attributes: int = 1
            Minimal number of flexible pairs.
            DEFAULT: 1
        max_stable_attributes: int = 5
            Maximal number of stable pairs.
            DEFAULT: 5
        max_flexible_attributes: int = 5
            Maximal number of flexible pairs.
            DEFAULT: 5
        is_strict_flexible : bool = True
            If true flexible attributes must be always actionable, if false they can also behave as stable attributes
            DEFAULT: True
        """
        if (self.action_rules):
            raise Exception("Fit was already called")
        self.stable_attributes = stable_attributes
        self.flexible_attributes = flexible_attributes
        self.consequent = consequent
        if bool(desired_classes) != bool(desired_changes):
            self.desired_state = DesiredState(desired_classes=desired_classes, desired_changes=desired_changes)
        else:
            raise Exception("Desired classes or desired changes must be entered")
        attributes = stable_attributes + flexible_attributes
        self._check_columns(attributes, consequent)
        stable = self.decisions.data[stable_attributes]
        flex = self.decisions.data[flexible_attributes]
        target = self.decisions.data[[consequent]]
        supp_df = self.decisions.data[[supp_col]]
        supp_series = supp_df.iloc[:, 0]
        supp = supp_series.tolist()
        conf_df = self.decisions.data[[conf_col]]
        conf_series = conf_df.iloc[:, 0]
        conf = conf_series.tolist()
        reduced_tables = Reduction(stable, flex, target, self.desired_state, supp, conf, is_nan)
        if is_reduction:
            reduced_tables.reduce()
        self.action_rules = ActionRules(
            reduced_tables.stable_tables,
            reduced_tables.flexible_tables,
            reduced_tables.decision_tables,
            self.desired_state,
            self.decisions,
            reduced_tables.supp,
            reduced_tables.conf,
            is_nan,
            min_stable_attributes,
            min_flexible_attributes,
            max_stable_attributes,
            max_flexible_attributes,
            is_strict_flexible
        )
        self.action_rules.fit()

    def get_action_rules(self) -> list:
        """Get machine representations of action rules.

        The output is a list of action
        rules. Each action rule is a list where the first part is an action rule itself, and the second part is
        a tuple of (support before, support after, action rule support), (confidence before, confidence after, action
        rule confidence) and uplift.

        Returns
        -------
        list
            Returns list of action rules.
        """
        return self.action_rules.action_rules

    def get_pretty_action_rules(self) -> list:
        """Get human-readable representations of action rules.

        Returns
        -------
        list
            Returns list of action rules.
        """
        if len(self.action_rules.action_rules_pretty_text) == 0:
            self.action_rules.pretty_text()
        return self.action_rules.action_rules_pretty_text

    def get_action_rules_representation(self) -> list:
        """Get math representation of action rules.

        Returns
        -------
        list
            Returns list of action rules.
        """
        if len(self.action_rules.action_rules_representation) == 0:
            self.action_rules.representation()
        return self.action_rules.action_rules_representation

    def get_source_data_for_ar(self, action_r_number: int, is_before: bool) -> pd.DataFrame:
        """ Get data frame with values which the action rule is based on.

        Yellow background - stable attributes
        Orange background - flexible attributes
        Red text - Target attribute, undesired state
        Green text - Target attribute, desired state

        Parameters
        ----------
        action_r_number : int
            The number of action rule - you can figure out from get_action_rules
        is_before : bool
            True shows instances in data that match the conditions of the "before" part of the action rule.
            False show instances in data that match the conditions of the "after" part of the action rule.

        Returns
        -------
        pd.DataFrame
            Returns data frame with transactions data.
        """
        if len(self.decisions.transactions) == 0:
            return pd.DataFrame()
        if is_before:
            classification = self.action_rules.classification_before[action_r_number]
        else:
            classification = self.action_rules.classification_after[action_r_number]
        decision = self.decisions.decision_table.loc[
            classification, self.stable_attributes + self.flexible_attributes]
        source_table = self._reduce_table_source(decision, self.decisions.data)
        return source_table.style.applymap(lambda x: 'background-color: yellow',
                                           subset=self.stable_attributes) \
            .applymap(lambda x: 'background-color: orange',
                      subset=self.flexible_attributes) \
            .applymap(lambda x: 'color: green' if x in self.desired_state.get_destination_classes() else 'color: red',
                      subset=[self.consequent])

    @staticmethod
    def _reduce_table_source(decision: pd.Series, source_table: pd.DataFrame) -> pd.DataFrame:
        """ Get data frame limited by concrete classification rule.

        Parameters
        ----------
        decision : pd.Series
            A classification rule.
        source_table : pd.DataFrame
            A source data frame.

        Returns
        -------
        pd.DataFrame
            Returns a limited data frame.
        """
        new_data = source_table.applymap(str).copy()
        for key, value in decision.items():
            if str(value).lower() != "nan":
                mask = new_data[key] == value
                new_data = new_data[mask]
        return new_data

    def predict(self, source_table: pd.DataFrame) -> pd.DataFrame:
        """ Predicts if any values would need to change their state.

        Parameters
        ----------
        source_table : pd.DataFrame
            A data frame with new observations.

        Returns
        -------
        pd.DataFrame
            Returns a data frame with recommended actions.
        """
        i = 0
        full_predicted_table = pd.DataFrame()
        for classification_before in self.action_rules.classification_before:
            classification_after = self.action_rules.classification_after[i]
            decision_before = self.decisions.decision_table.loc[
                classification_before, self.stable_attributes + self.flexible_attributes]
            decision_after = self.decisions.decision_table.loc[
                classification_after, self.stable_attributes + self.flexible_attributes]
            predicted_table = self._reduce_table_source(decision_before, source_table)
            if len(predicted_table.index) > 0:
                for key, value in decision_after.items():
                    if str(value).lower() != "nan" and key in self.flexible_attributes:
                        column = key + self.RECOMMENDED
                        predicted_table[column] = [value] * len(predicted_table.index)
                        predicted_table[self.ACTION_RULE] = [i] * len(predicted_table.index)
                        predicted_table = predicted_table.astype({self.ACTION_RULE: int})
                predicted_table[self.ACTION_RULE_TARGET] = \
                    [self.action_rules.action_rules[i][0][2][1][1]] * len(predicted_table.index)
                predicted_table[self.SUPPORT_BEFORE] = \
                    [self.action_rules.action_rules[i][1][0]] * len(predicted_table.index)
                predicted_table[self.SUPPORT_AFTER] = \
                    [self.action_rules.action_rules[i][1][1]] * len(predicted_table.index)
                predicted_table[self.ACTION_RULE_SUPPORT] = \
                    [self.action_rules.action_rules[i][1][2]] * len(predicted_table.index)
                predicted_table[self.CONFIDENCE_BEFORE] = \
                    [self.action_rules.action_rules[i][2][0]] * len(predicted_table.index)
                predicted_table[self.CONFIDENCE_AFTER] = \
                    [self.action_rules.action_rules[i][2][1]] * len(predicted_table.index)
                predicted_table[self.ACTION_RULE_CONFIDENCE] = \
                    [self.action_rules.action_rules[i][2][2]] * len(predicted_table.index)
                predicted_table[self.ACTION_RULE_UPLIFT] = \
                    [self.action_rules.action_rules[i][3]] * len(predicted_table.index)
            full_predicted_table = pd.concat([full_predicted_table, predicted_table], sort=True)
            i += 1
        # New columns always in the end
        cols = full_predicted_table.columns.tolist()
        if len(cols)>0:
            cols.append(cols.pop(cols.index(self.ACTION_RULE)))
            cols.append(cols.pop(cols.index(self.ACTION_RULE_TARGET)))
            cols.append(cols.pop(cols.index(self.SUPPORT_BEFORE)))
            cols.append(cols.pop(cols.index(self.SUPPORT_AFTER)))
            cols.append(cols.pop(cols.index(self.ACTION_RULE_SUPPORT)))
            cols.append(cols.pop(cols.index(self.CONFIDENCE_BEFORE)))
            cols.append(cols.pop(cols.index(self.CONFIDENCE_AFTER)))
            cols.append(cols.pop(cols.index(self.ACTION_RULE_CONFIDENCE)))
            cols.append(cols.pop(cols.index(self.ACTION_RULE_UPLIFT)))
            full_predicted_table = full_predicted_table[cols]
        return full_predicted_table
