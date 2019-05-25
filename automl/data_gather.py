from os import path
import pandas as pd

from automl import AutoML
from model_selection.problem_classification import ProblemClassifier

REGRESSION_FOLDER = 'data/regression'
CLASSIFICATION_FOLDER = 'data/classification'

CLASSIFICATION_TASKS = [
    ('digit-recognizer', 'label', ','),
    ('iris', 'Species', ','),
    ('mushroom-classification', 'class', ','),
    ('titanic', 'Survived', ','),
    ('predicting-a-pulsar-star', 'target_class', ','),
    ('optical-interconnection-network', 'Spatial Distribution', ','),
]

REGRESSION_TASKS = [
    ('house-prices-advanced-regression-techniques', 'SalePrice', ','),
    ('bike-sharing-day', 'cnt', ','),
    ('bike-sharing-hour', 'cnt', ','),
    ('forest-fires', 'area', ','),
    ('wine-quality-white', 'quality', ';'),
    ('wine-quality-red', 'quality', ';'),
    ('absenteeism-at-work', 'Absenteeism time in hours', ';'),
    ('automobiles', 'price', ','),

]


def collect(tasks, problem_type, stats_path, tasks_folder):
    for folder, target, delim in tasks:
        print('Starting', folder)
        data_file = path.join(tasks_folder, folder, 'train.csv')

        train = pd.read_csv(data_file, delimiter=delim)
        if train.shape[0] > 1000:
            train = train.sample(1000)

        X, y = train.drop(target, axis=1), train[target]

        auto_ml = AutoML(problem_type=problem_type)
        auto_ml.fit(X, y)
        auto_ml.describe()
        auto_ml.save_meta_data(stats_path, folder)


if __name__ == '__main__':
    collect(CLASSIFICATION_TASKS, ProblemClassifier.CLASSIFICATION, path.join(CLASSIFICATION_FOLDER, 'stats.csv'),
            CLASSIFICATION_FOLDER)
    collect(REGRESSION_TASKS, ProblemClassifier.REGRESSION, path.join(REGRESSION_FOLDER, 'stats.csv'),
            REGRESSION_FOLDER)
