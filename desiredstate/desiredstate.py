from typing import List
import pandas as pd


class DesiredState:
    """
    Desired state
    """

    def __init__(self, desired_classes: List[str] = None, desired_changes: List[list] = None):
        """
        Initialise the desired state. There are 2 options:
        1) Desired class or desired classes
        2) Desired changes
        """
        self.desired_classes = desired_classes
        self.desired_changes = desired_changes

    def is_candidate_couple(self, decision_column: pd.DataFrame) -> bool:
        """
        Is it possible to get any action rules?
        """
        # decision values are all the same
        if not self.has_variability(decision_column):
            return False
        # does have desired class
        if self.desired_classes and not self.is_desired_classes_candidate_couple(decision_column):
            return False
        # does have desired changes
        if self.desired_changes and not self.is_desired_changes_candidate_couple(decision_column):
            return False
        # otherwise
        return True

    def is_desired_classes_candidate_couple(self, decision_column: pd.DataFrame) -> bool:
        """
        Table has at least one desired class
        """
        is_candidate = False
        after = decision_column.iat[1, 0]
        if after in self.desired_classes:
            is_candidate = True
        return is_candidate

    def is_desired_changes_candidate_couple(self, decision_column: pd.DataFrame) -> bool:
        """
        Table has at least one desired class
        """
        is_candidate = False
        before = decision_column.iat[0, 0]
        after = decision_column.iat[1, 0]
        for desired_change in self.desired_changes:
            if desired_change[0] == before and desired_change[1] == after:
                is_candidate = True
        return is_candidate

    def is_candidate(self, decision_column: pd.DataFrame) -> bool:
        """
        Is it possible to get any action rules?
        """
        # decision values are all the same
        if not self.has_variability(decision_column):
            return False
        # does have desired class
        if self.desired_classes and not self.is_desired_classes_candidate(decision_column):
            return False
        # does have desired class
        if self.desired_changes and not self.is_desired_changes_candidate(decision_column):
            return False
        # otherwise
        return True

    def is_desired_classes_candidate(self, decision_column: pd.DataFrame) -> bool:
        """
        Table has at least one desired class
        """
        is_candidate = False
        unique_decisions = self.get_unique_values(decision_column, 0)
        for desired_class in self.desired_classes:
            if desired_class in unique_decisions:
                is_candidate = True
        return is_candidate

    def is_desired_changes_candidate(self, decision_column: pd.DataFrame) -> bool:
        """
        Table has at least one desired class
        """
        is_candidate = False
        unique_decisions = self.get_unique_values(decision_column, 0)
        for desired_change in self.desired_changes:
            if desired_change[0] in unique_decisions and desired_change[1] in unique_decisions:
                is_candidate = True
        return is_candidate

    def has_variability(self, decision_column: pd.DataFrame) -> bool:
        """
        Variability in decisions
        """
        unique_decisions = self.get_unique_values(decision_column, 0)
        if len(unique_decisions) == 1:
            return False
        return True

    def get_destination_classes(self) -> List[str]:
        destination_classes = []
        if self.desired_classes:
            destination_classes = destination_classes + self.desired_classes
        if self.desired_changes:
            for desired_change in self.desired_changes:
                destination_classes.append(desired_change[1])
        return destination_classes

    @staticmethod
    def get_unique_values(table: pd.DataFrame, column_number: int) -> list:
        """
        Get unique values from column
        """
        return table.iloc[:, column_number].unique()
