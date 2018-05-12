from sklearn.pipeline import TransformerMixin


class ConstantColumnsDrop(TransformerMixin):

    def __init__(self):
        self.columns_to_drop = []

    def fit(self, X, y=None, **kwargs):
        self.columns_to_drop = X.loc[:, (X == X.iloc[0]).all()].columns
        print(self.columns_to_drop)
        return self

    def transform(self, X, **kwargs):
        return X.drop(self.columns_to_drop, axis=1)

    def __repr__(self):
        return 'Constant columns: {}'.format(list(self.columns_to_drop))
