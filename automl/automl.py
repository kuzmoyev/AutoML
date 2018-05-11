from sklearn.pipeline import Pipeline

from preprocessing.preprocessing import Preprocessing
import pandas as pd


class AutoML:
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'

    def __init__(self,
                 max_time=600,
                 problem_type=None,
                 validation_size=0.25,
                 ordinal_features=None,
                 dimensionality_reduction=None,
                 log_pipeline=None,
                 initial_preprocessing_pipeline=None):
        """

        :param max_time: time limit for processing in seconds
        :param problem_type: class of the problem (CLASSIFICATION/REGRESSION)
        :param validation_size: fraction of data used for validation
        :param ordinal_features: dictionary in format {feature_name: [labels_in_order]}
        :param dimensionality_reduction: weather to use dimensionality reduction
        :param log_pipeline: output file for resulting pipeline description
        :param initial_preprocessing_pipeline: custom preprocessing pipeline used at the beginning of the resulting pipeline
        """
        self.ordinal_features = ordinal_features
        self.initial_preprocessing_pipeline = initial_preprocessing_pipeline
        self.pipeline = None

    def fit(self, X, y=None):
        steps = []

        # Initial pipeline
        if self.initial_preprocessing_pipeline:
            steps.append(('Initial pipeline', self.initial_preprocessing_pipeline))
            X = self.initial_preprocessing_pipeline.fit_transform(X, y)

        # Preprocessing pipeline
        preprocessing_pipeline = Preprocessing(ordinal_features=self.ordinal_features).get_pipeline(X, y)
        steps.append(('Preprocessing pipeline', preprocessing_pipeline))
        X = preprocessing_pipeline.fit_transform(X, y)

        self.pipeline = Pipeline(steps)

        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def predict(self, X):
        return self.pipeline.predict(X)


def main():
    df = pd.DataFrame({'A': [2, 1, 2, None, None],
                       'B': ['a', 'b', 'c', 'c', 'b'],
                       'C': [15, None, None, 16, 15],
                       'D': ['one', 'three', 'two', 'two', 'three']})

    print(df)
    model = AutoML(ordinal_features={'D': ['one', 'two', 'three']}).fit(df)

    df = pd.DataFrame({'A': [2, 1, None, None],
                       'B': ['b', 'c', 'c', 'b'],
                       'C': [None, None, 16, 15],
                       'D': ['three', 'two', 'two', 'five']})
    print(model.transform(df))


if __name__ == '__main__':
    main()
