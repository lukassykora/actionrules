import fim
import pandas as pd
import numpy as np
from typing import List


class Decisions:
    """
    The class Decision is used for classification rules mining. It uses library PyFim. A user
    can get the classification rules in a different way. In this case, the Decision class just holds
    classification rules.

    ...

    Attributes
    ----------
    data : pd.DataFrame
        Source transaction data.
    transactions : list
        Transactions ready for PyFIM.
    appearance : set
        Set of columns ready for PyFIM.
    rules : tuple
        Classification rules from PyFIM.
    decision_table : pd.DataFrame
        Classification rules in Pandas data frame.
    support : list
        List of supports.
    confidence : list
        List of confidences.
    max_length : int
        Max length of classification rules.
    Methods
    -------
    read_csv(self, file: str, **kwargs)
        Get transaction data from a CSV file.
    load_pandas(self, data_frame: pd.DataFrame)
        Get transaction data from a Pandas data frame.
    prepare_data_fim(self, antecedent_attributes: List[str], consequent: str)
        Transform data to be usable in PyFIM.
    fit_fim_apriori(self, conf: float=70, support: float=10)
        Train the model with PyFIM.
    generate_decision_table(self)
        Generate classification rules from the model.
    """

    def __init__(self):
        """Initialise.
        """
        self.data = pd.DataFrame()
        self.transactions = []
        self.appearance = set()
        self.rules = ()
        self.decision_table = pd.DataFrame()
        self.support = []
        self.confidence = []
        self.max_length = 10

    def read_csv(self, file: str, **kwargs):
        """Loads a data from a CSV file. It uses the Pandas read_csv method.

        Parameters
        ----------
        file : str
            The path to transaction data.
        **kwargs :
            Arbitrary keyword arguments (the same as in Pandas).
        """
        self.data = pd.read_csv(file, **kwargs)
        self.data = self.data.applymap(str)

    def load_pandas(self, data_frame: pd.DataFrame):
        """Loads a data from a Pandas data frame.

        Parameters
        ----------
        data_frame : pd.DataFrame
            Data frame with transaction data.
        """
        self.data = data_frame
        self.data = self.data.applymap(str)

    def prepare_data_fim(self, antecedent_attributes: List[str], consequent: str):
        """Data preparation for PyFIM.

        Parameters
        ----------
        antecedent_attributes : List[str]
            Antecedent columns names.
        consequent : str
            Consequent column name.
        """
        self.max_length = len(antecedent_attributes) + 1
        for index, row in self.data.iterrows():
            transaction_row = []
            for i, v in row.items():
                side_type = None
                if i == consequent:
                    side_type = "c"
                elif i in antecedent_attributes and str(v) != 'nan':
                    side_type = "a"
                if side_type:
                    self.appearance.add((str(i) + "<:> " + str(v), side_type))
                    transaction_row.append(str(i) + "<:> " + str(v))
            self.transactions.append(transaction_row)

    def fit_fim_apriori(self, conf: float=70, support: float=10):
        """Train the model to be able to get classification rules (PyFIM).

        Parameters
        ----------
        conf : float = 70
            Confidence.
            DEFAULT: 70%
        support : float = 10
            Support.
            DEFAULT: 10%
        """
        self.rules = fim.arules(self.transactions,
                                supp=support,
                                conf=conf,
                                report="sc",
                                mode="o",
                                appear=dict(self.appearance),
                                zmin=2,  # At least one antecedent and consequent
                                zmax=self.max_length)

    def generate_decision_table(self):
        """Generates table of classification rules.

        """
        decisions = {}
        for i, rule in enumerate(self.rules):
            values = {}
            # Antecedent
            for cedent in rule[1]:
                col = cedent.split("<:> ")[0]
                value = cedent.split("<:> ")[1]
                if value.lower() == "nan":
                    values[col] = np.NaN
                else:
                    values[col] = value
            # Subsequent
            col = rule[0].split("<:> ")[0]
            value = rule[0].split("<:> ")[1]
            if value.lower() == "nan":
                values[col] = np.NaN
            else:
                values[col] = value
            # Add support and confidence
            self.support.append(rule[2])
            self.confidence.append(rule[3])
            # Add row
            decisions[i] = values
        # Dictionary to DataFrame
        self.decision_table = pd.DataFrame(decisions).T

