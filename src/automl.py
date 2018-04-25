class AutoML:
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'

    def __init__(self,
                 max_time=600,
                 problem_type=None,
                 validation_size=0.25,
                 fillna_strategy=None,
                 ordinal_features=None,
                 dimensionality_reduction=None,
                 log_pipeline=None,
                 preprocessing_pipeline=None):
        """

        :param max_time: time limit for processing in seconds
        :param problem_type: class of the problem (CLASSIFICATION/REGRESSION)
        :param validation_size: fraction of data used for validation
        :param fillna_strategy: strategy for filling empty values
        :param ordinal_features: dictionary in format {feature_name: [labels_in_order]}
        :param dimensionality_reduction: weather to use dimensionality reduction
        :param log_pipeline: output file for resulting pipeline description
        :param preprocessing_pipeline: custom preprocessing pipeline used at the beginning of the resulting pipeline
        """

        preprocessing_pipeline = preprocessing_pipeline or []

    def build_model(self, data):
        # Should return pipeline.
        # TODO Return y column name(s)?
        pass
