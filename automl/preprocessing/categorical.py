from sklearn.pipeline import TransformerMixin
import numpy as np
import pandas as pd


class CategoricalToNumericalEncoder(TransformerMixin):

    def __init__(self, ordinal=None):
        self.nominal = None
        self.one_hot_columns = None
        self.ordinal = ordinal or {}

        # Make mappers
        for name, values in self.ordinal.items():
            self.ordinal[name] = {v: i for i, v in enumerate(values)}

    def fit(self, X, y=None, **kwargs):
        categorical_columns = set(X.select_dtypes(exclude=[np.number]).columns)
        nominal_names = categorical_columns - set(self.ordinal.keys())

        self.nominal = nominal_names
        self.one_hot_columns = pd.get_dummies(X, prefix=nominal_names, columns=nominal_names).columns

        return self

    def transform(self, X, **kwargs):
        # Converting ordinal
        for name, mapper in self.ordinal.items():
            dtype = np.int8 if len(mapper) < np.iinfo(np.int8).max else np.int32
            X[name] = X[name].map(mapper).fillna(-1).astype(dtype)

        # Converting nominal
        X = pd.get_dummies(X, prefix=self.nominal, columns=self.nominal)
        for column in self.one_hot_columns:
            if column not in X.columns:
                X[column] = 0

        return X

    def __repr__(self):
        return 'Ordinal: {}, Nominal: {}'.format(list(self.ordinal.keys()), list(self.nominal))
