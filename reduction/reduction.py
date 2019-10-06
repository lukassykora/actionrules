import pandas as pd
import numpy as np
from desiredstate import DesiredState


class Reduction:
    """
    Create reduction tables to limit the number of decision rules
    """

    def __init__(self, stable_columns: pd.DataFrame, flexible_columns: pd.DataFrame, decision_column: pd.DataFrame,
                 desired_state: DesiredState, supp: float, conf: float):
        """
        Initialise the decision table
        """
        self.stable_tables = [stable_columns]
        self.flexible_tables = [flexible_columns]
        self.decision_tables = [decision_column]
        self.stable_columns_count = self.get_columns_count(stable_columns)
        self.desired_state = desired_state
        self.supp = [pd.Series(supp)]
        self.conf = [pd.Series(conf)]

    @staticmethod
    def get_columns_count(columns: pd.DataFrame) -> int:
        """
        Get the number of columns
        """
        return len(columns.columns)

    @staticmethod
    def get_unique_values(table: pd.DataFrame, column_number: int) -> list:
        """
        Get unique values from column
        """
        return table.iloc[:, column_number].unique()

    def split_tables(self, stable_columns: pd.DataFrame, flexible_columns: pd.DataFrame, decision_column: pd.DataFrame,
                     split_position: int, supp_series: pd.Series, conf_series: pd.Series):
        """
        Split table by stable column
        """
        unique_values = self.get_unique_values(stable_columns, split_position)
        for unique_value in unique_values:
            mask = np.logical_or(stable_columns.iloc[:, split_position] == unique_value,
                                 stable_columns.iloc[:, split_position] == np.nan)
            new_stable_table = stable_columns[mask]
            new_flexible_table = flexible_columns[mask]
            new_decision_table = decision_column[mask]
            new_supp_series = supp_series[mask]
            new_conf_series = conf_series[mask]
            if self.desired_state.is_candidate(new_decision_table):
                self.stable_tables.append(new_stable_table)
                self.flexible_tables.append(new_flexible_table)
                self.decision_tables.append(new_decision_table)
                self.supp.append(new_supp_series)
                self.conf.append(new_conf_series)
            else:
                pass

    def reduce(self):
        """
        Reduce Decision table to many reduction tables
        """
        for split_position in range(self.stable_columns_count):
            for table in range(len(self.stable_tables)):
                stable_columns = self.stable_tables.pop(0)
                flexible_columns = self.flexible_tables.pop(0)
                decision_column = self.decision_tables.pop(0)
                supp_series = self.supp.pop(0)
                conf_series = self.conf.pop(0)
                self.split_tables(stable_columns, flexible_columns, decision_column, split_position, supp_series,
                                  conf_series)
