# Automated machine learning

Simple automated machine learning library based on relative landmarks described in *Sampling-Based Relative Landmarks: Systemat-
ically Test-Driving Algorithms Before Choosing* by Carlos Soares, Johann Petrak, and Pavel Brazdil.


## Usage

    from automl import AutoML
    
    ...
    
    auto_ml = AutoML(
        max_time=30,
        problem_type='regression'
    )
    auto_ml.fit(X_train, y_train)
    predictions = auto_ml.predict(X_test)


Describe built pipeline:

    auto_ml.describe()