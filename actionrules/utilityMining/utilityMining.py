import pandas as pd
from typing import List


class UtilityMining:
    """
    The class UtilityMining
    """

    def __init__(self,
                 utility_source,
                 min_util_dif: float
                 ):

        self.utility_function = None
        self.utility_table = None
        if isinstance(utility_source, pd.DataFrame):
            self.utility_table = utility_source
        elif callable(utility_source):
            self.utility_function = utility_source
        self.min_util_dif = min_util_dif

    def _check_utility(self, utility):
        if utility >= 0:
            return utility
        print('Warning - utility cannot be negative - negative value have been replaced by 0.')
        return 0

    def _get_utility(self, **kwargs):
        if callable(self.utility_function):
            utility = 0
            for key, value in kwargs.items():
                param = {}
                param[key] = value
                utility += self._check_utility(self.utility_function(**param))
            return utility
        if isinstance(self.utility_table, pd.DataFrame):
            utility = 0
            for key, value in kwargs.items():
                index = key + '_' + value
                try:
                    utility += self._check_utility(self.utility_table.at[index, 1])
                except KeyError:
                    print('Warning - key error at index ', index)
            return utility
        return 0

    def calculate_utilities(self, flex: List[pd.DataFrame], target: List[pd.DataFrame]):
        # calculating utilities
        utilities = []
        for i in range(len(flex)):
            params = {}
            for col in flex.columns:
                val = str(flex.at[i, col]).lower()
                if val != 'nan':
                    params[col] = val
            col = target.columns[0]
            val = target.at[i, col].lower()
            params[col] = val

            util = self._get_utility(**params)
            utilities.append(float(util))
        return utilities
