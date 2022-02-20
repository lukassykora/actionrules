import pandas as pd
from typing import List


class UtilityMining:
    """
    The class UtilityMining
    """

    def __init__(self,
                 min_util: float,
                 utility_source
                 ):

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
                except KeyError:
                    pass
            return utility
        return 0

    def fit(self, rules: List):
        """

        Returns
        -------

        """
        new_rules = []
        for rule in rules:
            params = {}
            for attr in (rule[0],) + rule[1]:
                name, value = attr.split('<:> ')
                if name and value:
                    params[name] = str(value).lower()  # výsledné action rules mají hodnoty v lower case

            utility = self._get_utility(**params)
            if utility >= self.min_util:
                new_rules.append(rule)
        return new_rules

    def utility_difference(self, rules: List):
        """
        To final action rules adds change in utility.
        -------

        """
        for rule in rules:
            params1 = {}
            params2 = {}

            for attr in rule[0][1]:
                name = attr[0]
                values = attr[1]
                params1[name] = values[0]
                params2[name] = values[1]

            name, values = rule[0][2]
            params1[name] = values[0]
            params2[name] = values[1]

            utility_before = self._get_utility(**params1)
            utility_after = self._get_utility(**params2)

            utility_dif = utility_after - utility_before
            rule.append((utility_dif, utility_before, utility_after))

    def sort_by_utility(self, rules: List):
        """
        Sorts action rules by difference in utility.

        Parameters
        ----------
        rules

        Returns
        -------

        """
        rules.sort(key=lambda rule: rule[4][0], reverse=True)