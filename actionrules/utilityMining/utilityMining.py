import pandas as pd
from typing import List


class UtilityMining:
    """
    The class UtilityMining
    """

    def __init__(self,
                 input_rules: List,
                 min_util: float,
                 utility_source
                 ):

        self.input_rules = input_rules
        self.utility_function = None
        self.utility_table = None
        if isinstance(utility_source, pd.DataFrame):
            self.utility_table = utility_source
        elif callable(utility_source):
            self.utility_function = utility_source
        self.min_util = min_util

    def _get_utility(self, **kwargs):
        if callable(self.utility_function):
            return self.utility_function(**kwargs)
        if isinstance(self.utility_table, pd.DataFrame):
            utility = 0
            for key, value in kwargs.items():
                index = key + '_' + value
                try:
                    utility += self.utility_table.at[index, 1]
                finally:
                    pass
            return utility
        return 0

    def fit(self):
        """

        Returns
        -------

        """
        new_rules = []
        for rule in self.input_rules:
            params = {}
            for attr in rule[1]:
                name, value = attr.split('<:> ')
                if name and value:
                    params[name] = value

            utility = self._get_utility(**params)
            if utility >= self.min_util:
                new_rules.append(rule)
        return new_rules
