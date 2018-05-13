from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


class Scaler(TransformerMixin):

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        self.scaler.fit(X[numerical_columns], y)
        return self

    def transform(self, X):
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        X = X.copy()
        X[numerical_columns] = self.scaler.transform(X[numerical_columns])
        return X

    def __repr__(self):
        return self.scaler.__repr__()
