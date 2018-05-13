from sklearn.pipeline import Pipeline
import numpy as np

from preprocessing.categorical import CategoricalToNumericalEncoder
from preprocessing.drop_constant import ConstantColumnsDrop
from preprocessing.imputation import AutoImputer
from preprocessing.scaler import Scaler


class Preprocessing:
    def __init__(self,
                 ordinal_features=None,
                 dimensionality_reduction_strategy=None):
        """

        :param ordinal_features: dictionary in format {feature_name: [labels_in_order]}
        :param dimensionality_reduction_strategy: strategy for dimensionality reduction
        """

        self.ordinal_features = ordinal_features
        self.dimensionality_reduction_strategy = dimensionality_reduction_strategy

    def get_pipeline(self, X, y=None):
        steps = []

        # Dropping constant columns
        if (X == X.iloc[0]).all().any():
            dropper = ConstantColumnsDrop()
            steps.append(('Constant columns drop', dropper))

        # Filling missing values
        if X.isnull().values.any():
            imputer = AutoImputer()
            steps.append(('Imputer', imputer))

        # Scaling numerical values
        if not X.select_dtypes(include=[np.number]).columns.empty:
            scaler = Scaler()
            steps.append(('Scaler', scaler))

        # Converting categorical features to numerical
        if not X.select_dtypes(exclude=[np.number]).columns.empty:
            converter = CategoricalToNumericalEncoder(ordinal=self.ordinal_features)
            steps.append(('Categorical converter', converter))

        return Pipeline(steps) if steps else None
