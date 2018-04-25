class Preprocessing:
    def __init__(self,
                 fillna_strategy=None,
                 ordinal_features=None,
                 dimensionality_reduction=None,
                 log_pipeline=None,
                 preprocessing_pipeline=None):
        """

        :param fillna_strategy: strategy for filling empty values
        :param ordinal_features: dictionary in format {feature_name: [labels_in_order]}
        :param dimensionality_reduction: weather to use dimensionality reduction
        :param log_pipeline: output file for resulting pipeline description
        :param preprocessing_pipeline: custom preprocessing pipeline used at the beginning of the resulting pipeline
        """

        preprocessing_pipeline = preprocessing_pipeline or []

    def get_pipeline(self, data):
        # TODO don't forget preprocessing pipeline
        pass
