from sklearn.pipeline import Pipeline
import pandas as pd

from preprocessing.preprocessing import Preprocessing
from model_selection.problem_classification import ProblemClassifier
from model_selection.metadata_extraction import get_extractor


class AutoML:

    def __init__(self,
                 max_time=600,
                 problem_type=ProblemClassifier.REGRESSION,
                 validation_size=0.25,
                 ordinal_features=None,
                 dimensionality_reduction=None,
                 initial_preprocessing_pipeline=None):
        """

        :param max_time: time limit for processing in seconds
        :param problem_type: class of the problem (CLASSIFICATION/REGRESSION)
        :param validation_size: fraction of data used for validation
        :param ordinal_features: dictionary in format {feature_name: [labels_in_order]}
        :param dimensionality_reduction: weather to use dimensionality reduction
        :param initial_preprocessing_pipeline: custom preprocessing pipeline used at the beginning of the resulting pipeline
        """
        self.max_time = max_time
        self.problem_type = problem_type
        self.validation_size = validation_size
        self.ordinal_features = ordinal_features
        self.dimensionality_reduction = dimensionality_reduction
        self.initial_preprocessing_pipeline = initial_preprocessing_pipeline

        self.pipeline = None
        self.meta_extractor = get_extractor(problem_type)

    def fit(self, X, y):
        steps = []

        # Initial pipeline
        if self.initial_preprocessing_pipeline:
            steps.append(('Initial pipeline', self.initial_preprocessing_pipeline))
            X = self.initial_preprocessing_pipeline.fit_transform(X, y)

        self.meta_extractor.extract_initial(X, y)

        # Preprocessing pipeline
        preprocessing_pipeline = Preprocessing(ordinal_features=self.ordinal_features).get_pipeline(X, y)
        steps.append(('Preprocessing pipeline', preprocessing_pipeline))
        X = preprocessing_pipeline.fit_transform(X, y)

        self.pipeline = Pipeline(steps)

        self.meta_extractor.extract_preprocessed(X, y)
        print(self.meta_extractor.as_df().T)

        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def predict(self, X):
        return self.pipeline.predict(X)

    def describe(self):
        print(self.pipeline)


def main_():
    X = pd.DataFrame({'A': [2, 1, 2, None, None, 2],
                      'B': ['a', 'b', 'c', 'c', 'b', 'a'],
                      'C': [15, None, None, 16, 15, 22],
                      'D': ['one', 'three', 'two', 'two', 'three', 'one']})
    y = pd.Series([2, 5, 3, 3, 5, 2])

    auto_ml = AutoML(ordinal_features={'D': ['one', 'two', 'three']})

    model = auto_ml.fit(X, y)

    X = pd.DataFrame({'A': [2, 1, None, None],
                      'B': ['b', 'c', 'c', 'b'],
                      'C': [None, None, 16, 15],
                      'D': ['three', 'two', 'two', 'five']})

    auto_ml.describe()


def main():
    df = pd.read_csv('data/house-prices-advanced-regression-techniques/train.csv')

    X, y = df.drop('SalePrice', axis=1), df['SalePrice']

    auto_ml = AutoML()
    model = auto_ml.fit(X, y)
    

if __name__ == '__main__':
    main()
