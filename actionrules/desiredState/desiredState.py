from typing import List
import pandas as pd


class DesiredState:
    """
    The class DesiredState is responsible for all features that are needed to recognize the
    candidate pair for the target variable.

    ...

    Attributes
    ----------
    desired_classes : List[str] or None
        Desired classes.
    desired_changes : List[list] or None
        Desired changes.

    Methods
    -------
    is_candidate_decision(self, decision_before: str, decision_after: str) -> bool
        It checks if a pair of consequent values is a candidate.
    is_candidate(self, decision_column: pd.DataFrame) -> bool
        Is it possible to get any action rules (variability, desired classes)?
    get_destination_classes(self) -> List[str]
        Get list of possible desired classes.
    get_not_in_default_classes
        Get the possible before part of consequent.
    """

    def __init__(self, desired_classes: List[str] = None, desired_changes: List[list] = None):
        """Initialise the desired state. There are 2 options:

        1) Desired class or desired classes
        2) Desired changes

        Parameters
        ----------
        desired_classes : List[str]
            List of desired classes. For example: ['survived', 'survived with injury'].
        desired_changes : List[list]
            Concrete desired changes. For example: [['death', 'survived'], ['death', 'survived with injury']]
        """
        self.desired_classes = self._candidates_to_string(desired_classes)
        self.desired_changes = self._candidates_to_string(desired_changes)

    @staticmethod
    def _candidates_to_string(desired_data: list) -> list:
        """It converts all values to strings.

        Parameters
        ----------
        desired_data : list
            Desired classes or changes.

        Returns
        -------
        list
            The same list with string values.
        """
        converted_data = []
        if isinstance(desired_data, list):
            for val in desired_data:
                if isinstance(val, list):
                    converted = [str(v) for v in val]
                else:
                    converted = str(val)
                converted_data.append(converted)
        return converted_data


    def is_candidate_decision(self, decision_before: str, decision_after: str) -> bool:
        """It checks if a pair of consequent is a candidate.

        Parameters
        ----------
        decision_before : str
            The value of a decision before.
        decision_after : str
            The value of a decision after.

        Returns
        -------
        bool
            Could it be a candidate pair?
        """
        if decision_before == decision_after:
            return False
        if self.desired_classes and decision_after not in self.desired_classes:
            return False
        if self.desired_changes and \
                [decision_before, decision_after] not in self.desired_changes:
            return False
        return True

    def is_candidate(self, decision_column: pd.DataFrame) -> bool:
        """Is it possible to get any action rules from a consequent data frame (variability, desired classes)?

        Parameters
        ----------
        decision_column : pd.DataFrame
            Data frame with consequent values.

        Returns
        -------
        bool
            Could action rules be in the data frame?
        """
        # decision values are all the same
        if not self._has_variability(decision_column):
            return False
        # does have desired class
        if self.desired_classes and not self._is_desired_classes_candidate(decision_column):
            return False
        # does have desired class
        if self.desired_changes and not self._is_desired_changes_candidate(decision_column):
            return False
        # otherwise
        return True

    def _is_desired_classes_candidate(self, decision_column: pd.DataFrame) -> bool:
        """Check if table has at least one desired class.

        Parameters
        ----------
        decision_column : pd.DataFrame
            Data frame with consequent values.

        Returns
        -------
        bool
            Does the data frame contain any desired classes?
        """
        is_candidate = False
        unique_decisions = self._get_unique_values(decision_column, 0)
        for desired_class in self.desired_classes:
            if desired_class in unique_decisions:
                is_candidate = True
        return is_candidate

    def _is_desired_changes_candidate(self, decision_column: pd.DataFrame) -> bool:
        """Check if table has at least one desired change of consequent.

        Parameters
        ----------
        decision_column : pd.DataFrame
            Data frame with consequent values.

        Returns
        -------
        bool
            Does the data frame contain any desired changes?
        """
        is_candidate = False
        unique_decisions = self._get_unique_values(decision_column, 0)
        for desired_change in self.desired_changes:
            if desired_change[0] in unique_decisions and desired_change[1] in unique_decisions:
                is_candidate = True
        return is_candidate

    def _has_variability(self, decision_column: pd.DataFrame) -> bool:
        """Is there enough variability in desired classes?

        Parameters
        ----------
        decision_column : pd.DataFrame
            Data frame with consequent values.

        Returns
        -------
        bool
            Does the data frame have enough variability?
        """
        unique_decisions = self._get_unique_values(decision_column, 0)
        if len(unique_decisions) == 1:
            return False
        return True

    def get_destination_classes(self) -> List[str]:
        """Get all possible desired classes.

        Returns
        -------
        List[str]
            All possible desired classes.
        """
        destination_classes = []
        if self.desired_classes:
            destination_classes = destination_classes + self.desired_classes
        if self.desired_changes:
            for desired_change in self.desired_changes:
                destination_classes.append(desired_change[1])
        return destination_classes

    def get_not_in_default_classes(self) -> List[str]:
        """Get the possible before part of consequent.

        Returns
        -------
        List[str]
            All desired classes that are not in before part.
        """
        destination_classes = self.get_destination_classes()
        if self.desired_changes:
            for desired_change in self.desired_changes:
                if desired_change[0] in destination_classes:
                    destination_classes.remove(desired_change[0])
        return destination_classes

    @staticmethod
    def _get_unique_values(table: pd.DataFrame, column_number: int) -> list:
        """Get unique values from column

        Parameters
        ----------
        table : pd.DataFrame
            Data frame with consequent values.
        column_number : int
            Column number to be checked,

        Returns
        -------
        list
            All unique values.
        """
        return table.iloc[:, column_number].unique()
