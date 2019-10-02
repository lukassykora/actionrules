import fim
import pandas as pd
import numpy as np


class Decisions:
    """
    Decisions
    """

    def __init__(self):
        """
        Initialise the empty dataframe.
        """
        self.data = pd.DataFrame()
        self.transactions = []
        self.appearance = set()
        self.rules = ()
        self.decision_table = pd.DataFrame()
        self.train_x = pd.DataFrame()
        self.train_y = pd.DataFrame()
        self.label_encoders = {}
        self.support = []
        self.confidence = []

    def read_csv(self, file, **kwargs):
        """
        Load data from csv.
        """
        self.data = pd.read_csv(file, **kwargs)

    def load_pandas(self, dataframe):
        """
        Load data from pandas dataframe.
        """
        self.data = dataframe

    def prepare_data_fim(self, antecedents, consequent):
        """
        Data preparation.
        """
        for index, row in self.data.iterrows():
            transaction_row = []
            for i, v in row.iteritems():
                side_type = None
                if i == consequent:
                    side_type = "c"
                elif i in antecedents:
                    side_type = "a"
                if side_type:
                    self.appearance.add((str(i) + ": " + str(v), side_type))
                    transaction_row.append(str(i) + ": " + str(v))
            self.transactions.append(transaction_row)

    def fit_fim_apriori(self, conf=70, support=10, minlen=2, maxlen=10):
        self.rules = fim.arules(self.transactions,
                                supp=support,
                                conf=conf,
                                report="sc",
                                mode="o",
                                appear=dict(self.appearance),
                                zmin=minlen,
                                zmax=maxlen)

    def generate_decision_table(self):
        for rule in self.rules:
            values = []
            cols = []
            # Antecdents
            for cedent in rule[1]:
                cols.append(cedent.split(": ")[0])
                value = cedent.split(": ")[1]
                if value == "nan":
                    values.append(np.NaN)
                else:
                    values.append(value)
            # Subsequent
            cols.append(rule[0].split(": ")[0])
            value = rule[0].split(": ")[1]
            if value == "nan":
                values.append(np.NaN)
            else:
                values.append(value)

            df2 = pd.DataFrame([values], columns=cols)
            self.decision_table = self.decision_table.append(df2, sort=False, ignore_index=True)
            self.support.append(rule[2])
            self.confidence.append(rule[3])
