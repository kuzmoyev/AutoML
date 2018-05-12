from sklearn.pipeline import Pipeline
import pandas as pd

from model_selection.model_selection import get_evaluator
from preprocessing.preprocessing import Preprocessing
from model_selection.problem_classification import ProblemClassifier
from model_selection.metadata_extraction import get_extractor


class AutoML:

    def __init__(self,
                 max_time=600,
                 time_accuracy_trade_rate=0.05,
                 problem_type=ProblemClassifier.REGRESSION,
                 ordinal_features=None,
                 dimensionality_reduction=None,
                 initial_preprocessing_pipeline=None):
        """

        :param max_time: time limit for processing in seconds
        :param time_accuracy_trade_rate: amount of accuracy(%) willing to trade for 10 times speed-up.
        :param problem_type: class of the problem (CLASSIFICATION/REGRESSION)
        :param ordinal_features: dictionary in format {feature_name: [labels_in_order]}
        :param dimensionality_reduction: weather to use dimensionality reduction
        :param initial_preprocessing_pipeline: custom preprocessing pipeline used at the beginning of the resulting pipeline
        """
        self.max_time = max_time
        self.problem_type = problem_type
        self.ordinal_features = ordinal_features
        self.dimensionality_reduction = dimensionality_reduction
        self.initial_preprocessing_pipeline = initial_preprocessing_pipeline

        self.pipeline = None
        self.meta_extractor = get_extractor(problem_type)
        self.model_evaluator = get_evaluator(problem_type, time_limit=max_time,
                                             time_accuracy_trade_rate=time_accuracy_trade_rate)

    def fit(self, X, y):
        steps = []

        # Initial pipeline
        if self.initial_preprocessing_pipeline:
            steps.append(('Initial pipeline', self.initial_preprocessing_pipeline))
            X = self.initial_preprocessing_pipeline.fit_transform(X, y)

        self.meta_extractor.extract_initial(X, y)

        # Preprocessing pipeline
        preprocessing_pipeline = Preprocessing(ordinal_features=self.ordinal_features).get_pipeline(X, y)
        if preprocessing_pipeline:
            steps.append(('Preprocessing pipeline', preprocessing_pipeline))
            X = preprocessing_pipeline.fit_transform(X, y)

        self.meta_extractor.extract_preprocessed(X, y)

        self.model_evaluator.evaluate_models(X, y, self.meta_extractor.as_dict())

        model = self.model_evaluator.get_best_model()
        model.fit(X, y)
        steps.append(('Model', model))

        self.pipeline = Pipeline(steps)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def predict(self, X):
        return self.pipeline.predict(X)

    def describe(self):
        self._describe_estimator(estimator=self.pipeline)

    def _describe_estimator(self, estimator, level=0):
        if isinstance(estimator, Pipeline):
            for name, estimator in estimator.steps:
                print('\t' * level, name, ':', sep='')
                self._describe_estimator(estimator, level + 1)
        else:
            print('\t' * level, estimator, sep='')


def main():
    from os import path
    import numpy as np

    folder = 'data/regression/house-prices-advanced-regression-techniques'

    df = pd.read_csv(path.join(folder, 'train.csv'))
    X_test = pd.read_csv(path.join(folder, 'test.csv'))

    X, y = df.drop('SalePrice', axis=1), df['SalePrice']

    auto_ml = AutoML(problem_type=ProblemClassifier.REGRESSION)
    auto_ml.fit(X, y)
    auto_ml.describe()

    predictions = auto_ml.predict(X_test)
    submission = pd.DataFrame({'Id': X_test.Id, 'SalePrice': predictions})
    submission.to_csv(path.join(folder, 'submission.csv'), index=False)


if __name__ == '__main__':
    main()
