import pandas as pd
import numpy as np

from actionrules.desiredState import DesiredState


class Reduction:
    """
    The class Reduction creates the Reduction tree that speed up the discovery process.
    This feature can be skipped. In this case, the Reduction class is not initialized.

    ...

    Attributes
    ----------
    stable_tables : List[pd.DataFrame]
        List of data frames with stable attributes.
    flexible_tables : List[pd.DataFrame]
        List of data frames with flexible attributes.
    decision_tables : List[pd.DataFrame]
        List of data frames with consequent.
    stable_columns_count : int
        Count of stable attributes.
    flexible_columns_count : int
        Count of flexible attributes.
    desired_state : DesiredState
        DesiredState object.
    self.supp : List[pd.Series]
        List of Pandas Series with support.
    self.conf : List[pd.Series]
        List of Pandas Series with confidence.
    self.is_nan : bool
        Should uncertainty be used?

    Methods
    -------
    reduce(self)
        Reduce Decision table to many tables.
    """

    def __init__(self, stable_columns: pd.DataFrame, flexible_columns: pd.DataFrame, decision_column: pd.DataFrame,
                 desired_state: DesiredState, supp: float, conf: float, is_nan: bool, util_flex: float = None,
                 util_target: float = None):
        """Initialise.

        Parameters
        ----------
        stable_columns: pd.DataFrame
            Data frame with stable attributes.
        flexible_columns: pd.DataFrame
            Data frame with flexible attributes.
        decision_column: pd.DataFrame
            Data frame with consequent.
        desired_state: DesiredState
            DesiredState object.
        supp: float
            Support, for example 0.1 means 10%.
        conf: float
            Confidence, for example 0.8 means 80%.
        is_nan: bool
            Should uncertainty be used?
        """
        self.stable_tables = [stable_columns]
        self.flexible_tables = [flexible_columns]
        self.decision_tables = [decision_column]
        self.stable_columns_count = self._get_columns_count(stable_columns)
        self.flexible_columns_count = self._get_columns_count(flexible_columns)
        self.desired_state = desired_state
        self.supp = [pd.Series(supp)]
        self.conf = [pd.Series(conf)]
        self.is_nan = is_nan
        self.util_flex = None
        if util_flex:
            self.util_flex = [pd.Series(util_flex)]
        self.util_target = None
        if util_target:
            self.util_target = [pd.Series(util_target)]

    @staticmethod
    def _get_columns_count(columns: pd.DataFrame) -> int:
        """Get the number of columns

        Parameters
        ----------
        columns: pd.DataFrame
            Data frame to check.

        Returns
        -------
        int
            Count of columns.
        """
        return len(columns.columns)

    @staticmethod
    def _get_unique_values(table: pd.DataFrame, column_number: int) -> list:
        """Get unique values from column

        Parameters
        ----------
        table : pd.DataFrame
            Data frame with consequent values.
        column_number : int
            Column number to be checked.

        Returns
        -------
        list
            All unique values.
        """
        return table.iloc[:, column_number].unique()

    def _split_tables_by_stable(self, stable_columns: pd.DataFrame, flexible_columns: pd.DataFrame,
                                decision_column: pd.DataFrame,
                                split_position: int, supp_series: pd.Series, conf_series: pd.Series,
                                util_flex_series: pd.Series = None, util_target_series: pd.Series = None):
        """Split table by stable column.

        Parameters
        ----------
        stable_columns : pd.DataFrame
            Stable attributes.
        flexible_columns : pd.DataFrame
            Flexible attributes.
        decision_column : pd.DataFrame
            Consequent column.
        split_position : int
            The position where the split can be made.
        supp_series : pd.Series
            Support column.
        conf_series : pd.Series
            Confidence column.
        """
        unique_values = self._get_unique_values(stable_columns, split_position)
        for unique_value in unique_values:
            if self.is_nan:
                mask = np.logical_or(stable_columns.iloc[:, split_position] == unique_value,
                                     np.logical_or(stable_columns.iloc[:, split_position].isnull(),
                                                   stable_columns.iloc[:, split_position] == 'nan'
                                                   )
                                     )
            else:
                if str(unique_value).lower() == "nan":
                    mask = np.logical_or(stable_columns.iloc[:, split_position].isnull(),
                                         stable_columns.iloc[:, split_position] == 'nan'
                                         )
                else:
                    mask = stable_columns.iloc[:, split_position] == unique_value
            new_stable_table = stable_columns[mask]
            new_flexible_table = flexible_columns[mask]
            new_decision_table = decision_column[mask]
            new_supp_series = supp_series[mask]
            new_conf_series = conf_series[mask]
            new_util_flex_series = None
            if util_flex_series is not None:
                new_util_flex_series = util_flex_series[mask]
            new_util_target_series = None
            if util_target_series is not None:
                new_util_target_series = util_target_series[mask]
            if self.desired_state.is_candidate(new_decision_table):
                self.stable_tables.append(new_stable_table)
                self.flexible_tables.append(new_flexible_table)
                self.decision_tables.append(new_decision_table)
                self.supp.append(new_supp_series)
                self.conf.append(new_conf_series)
                if new_util_flex_series is not None:
                    self.util_flex.append(new_util_flex_series)
                if new_util_target_series is not None:
                    self.util_target.append(new_util_target_series)

    def reduce(self):
        """Reduce Decision table to many reduction tables.

        """
        for split_position in range(self.stable_columns_count):
            for table in range(len(self.stable_tables)):
                stable_columns = self.stable_tables.pop(0)
                flexible_columns = self.flexible_tables.pop(0)
                decision_column = self.decision_tables.pop(0)
                supp_series = self.supp.pop(0)
                conf_series = self.conf.pop(0)
                util_flex_series = None
                if self.util_flex:
                    util_flex_series = self.util_flex.pop(0)
                util_target_series = None
                if self.util_target:
                    util_target_series = self.util_target.pop(0)
                self._split_tables_by_stable(stable_columns,
                                             flexible_columns,
                                             decision_column,
                                             split_position,
                                             supp_series,
                                             conf_series,
                                             util_flex_series,
                                             util_target_series
                                             )
