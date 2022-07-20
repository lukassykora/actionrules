import pandas as pd
from typing import List


class UtilityMining:
    """
    The class UtilityMining contains methods for calculating utility of classification rules
    This feature can be skipped. In this case, the UtilityMining class is not initialized.

    ...

    Attributes
    ----------
    utility_function : Function
        Function providing utility values.
    utility_table : pd.DataFrame
        DataFrame providing utility values.
    min_util_dif: float
        Number representing minimal desired change in utility caused by action.

    Methods
    -------
    calculate_utilities(self,
                        flex: List[pd.DataFrame],
                        target: List[pd.DataFrame]):
        Calculates utility for single classification rule.
    """

    def __init__(self,
                 utility_source,
                 min_util_dif: float,
                 min_profit: float
                 ):
        """Initialise.

        Parameters
        ----------
        utility_source : Function or pd.DataFrame
            Function or DataFrame providing utility values.
        min_util_dif: float
            Number representing minimal desired change in utility caused by action.
        min_profit: float
            Number representing minimal profit.
        """

        self.utility_function = None
        self.utility_table = None
        if isinstance(utility_source, pd.DataFrame):
            self.utility_table = utility_source
        elif callable(utility_source):
            self.utility_function = utility_source
        self.min_util_dif = min_util_dif
        self.min_profit = min_profit

    def _is_source_defined(self):
        return isinstance(self.utility_table, pd.DataFrame) or callable(self.utility_function)

    def use_utility_mining(self):
        return self.use_min_dif() or self.use_min_profit()

    def use_min_dif(self):
        return self.min_util_dif is not None and self._is_source_defined()

    def use_min_profit(self):
        return self.min_profit is not None and self._is_source_defined()

    # def _check_utility(self, utility):
    #     """ Checks if the utility values are nonnegative. If they are, they are returned,
    #         if they are not, 0 is returned and warning is printed out.
    #
    #     Parameters
    #     ----------
    #     utility : float
    #         Utility value that is currently being checked.
    #
    #     Returns
    #     -------
    #     float
    #         Returns particular utility value or 0.
    #     """
    #     if utility >= 0:
    #         return utility
    #     print('Warning - utility cannot be negative - negative value have been replaced by 0.')
    #     return 0

    def _get_utility(self, **kwargs):
        """Returns sum of utilities of input parameters, checks utility values and takes
            utility values from correct source.

        Parameters
        ----------
        **kwargs : Dictionary
            Dictionary of arguments and argument values of classification rule.

        Returns
        -------
        float
            Returns particular sum of utility values.
        """
        if callable(self.utility_function):
            utility = 0
            for key, value in kwargs.items():
                param = {}
                param[key] = value
                utility += self.utility_function(**param)
            return utility
        if isinstance(self.utility_table, pd.DataFrame):
            utility = 0
            for key, value in kwargs.items():
                index = key + '_' + value
                try:
                    utility += self.utility_table.at[index, 1]
                except KeyError:
                    print('Warning - key error at index ', index)
            return utility
        return 0

    def calculate_utilities(self, flex: List[pd.DataFrame], target: List[pd.DataFrame]):
        """For list of flexible and target attributes creates list of utilities.

        Parameters
        ----------
        flex: List[pd.DataFrame]
            List with flexible attributes.
        target: List[pd.DataFrame])
            List with target attributes.

        Returns
        -------
        List
            Returns list of utilities.
        """
        if not self._is_source_defined():
            raise Exception('No utility source defined')
        utilities_flex = []
        utilities_target = []
        for i in range(len(flex)):
            # utility of flexible attributes
            params = {}
            for col in flex.columns:
                val = str(flex.at[i, col]).lower()
                if val != 'nan':
                    params[col] = val
            util_flex = self._get_utility(**params)

            # utility of target attribute
            params = {}
            col = target.columns[0]
            val = target.at[i, col].lower()
            params[col] = val
            util_target = self._get_utility(**params)

            # adds tuple of flexible and target utilities to the return lists
            utilities_flex.append(float(util_flex))
            utilities_target.append(float(util_target))
        return utilities_flex, utilities_target

