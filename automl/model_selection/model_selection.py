from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, \
    RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcess, GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, \
    RidgeClassifier, RidgeClassifierCV, SGDClassifier, ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, \
    HuberRegressor, Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression, \
    MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV, OrthogonalMatchingPursuit, \
    OrthogonalMatchingPursuitCV, PassiveAggressiveRegressor, RANSACRegressor, Ridge, RidgeCV, SGDRegressor, \
    TheilSenRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier, KNeighborsRegressor, \
    RadiusNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import LinearSVC, NuSVC, SVC, LinearSVR, NuSVR, SVR
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor

import signal


class BaseEvaluator:
    models = None

    def __init__(self, time_limit=0):
        """
        :param time_limit: time limit for models evaluations. Default is 0 (no limit)
        """

        self.time_limit = time_limit

    def evaluate_models(self, X, y):
        try:
            signal.alarm(self.time_limit)

        except TimeLimitExceeded:
            pass
        else:
            signal.alarm(0)

    def _evaluate_model(self, model, X, y):
        # TODO return model stats (error, processing time...)
        pass


class ClassificationEvaluator(BaseEvaluator):
    models = [
        AdaBoostClassifier,
        BaggingClassifier,
        BernoulliNB,
        CalibratedClassifierCV,
        DecisionTreeClassifier,
        ExtraTreeClassifier,
        ExtraTreesClassifier,
        GaussianNB,
        GaussianProcessClassifier,
        GradientBoostingClassifier,
        KNeighborsClassifier,
        LabelPropagation,
        LabelSpreading,
        LinearDiscriminantAnalysis,
        LinearSVC,
        LogisticRegression,
        LogisticRegressionCV,
        MLPClassifier,
        MultinomialNB,
        NearestCentroid,
        NuSVC,
        PassiveAggressiveClassifier,
        Perceptron,
        QuadraticDiscriminantAnalysis,
        RadiusNeighborsClassifier,
        RandomForestClassifier,
        RidgeClassifier,
        RidgeClassifierCV,
        SGDClassifier,
        SVC
    ]


class RegressionEvaluator(BaseEvaluator):
    models = [
        ARDRegression,
        AdaBoostRegressor,
        BaggingRegressor,
        BayesianRidge,
        CCA,
        DecisionTreeRegressor,
        ElasticNet,
        ElasticNetCV,
        ExtraTreeRegressor,
        ExtraTreesRegressor,
        GaussianProcess,
        GaussianProcessRegressor,
        GradientBoostingRegressor,
        HuberRegressor,
        KNeighborsRegressor,
        KernelRidge,
        Lars,
        LarsCV,
        Lasso,
        LassoCV,
        LassoLars,
        LassoLarsCV,
        LassoLarsIC,
        LinearRegression,
        LinearSVR,
        MLPRegressor,
        MultiTaskElasticNet,
        MultiTaskElasticNetCV,
        MultiTaskLasso,
        MultiTaskLassoCV,
        NuSVR,
        OrthogonalMatchingPursuit,
        OrthogonalMatchingPursuitCV,
        PLSCanonical,
        PLSRegression,
        PassiveAggressiveRegressor,
        RANSACRegressor,
        RadiusNeighborsRegressor,
        RandomForestRegressor,
        Ridge,
        RidgeCV,
        SGDRegressor,
        SVR,
        TheilSenRegressor
    ]


class TimeLimitExceeded(Exception):
    pass


def handler(signum, frame):
    raise TimeLimitExceeded


signal.signal(signal.SIGALRM, handler)
