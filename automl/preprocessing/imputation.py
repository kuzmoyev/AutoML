from sklearn.pipeline import TransformerMixin
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np

BACK_FILL = 'bfill'
FRONT_FILL = 'ffill'
MEAN_FILL = 'mean'
MEDIAN_FILL = 'median'
MOST_FREQUENT_FILL = 'most_frequent'
KNN_FILL = 'knn'

SKIP = 'skip'


class AutoImputer(TransformerMixin):

    def __init__(self, strategy=MOST_FREQUENT_FILL, columns='all'):
        self.strategy = strategy
        self.columns = columns
        self.fill = None

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

    def __repr__(self):
        return 'Strategy: {}'.format(self.strategy)
