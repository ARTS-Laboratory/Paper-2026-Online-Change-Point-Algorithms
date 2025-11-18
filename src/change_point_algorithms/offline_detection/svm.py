from collections.abc import Sequence

import numpy as np
from change_point_algorithms.offline_detection.offline_model_helpers import (
    dense_to_time_intervals, estimator_prediction,
    format_single_feature_data)
from sklearn.svm import SVC


def get_svm_model(train_data: np.ndarray, train_labels: np.ndarray, time_vec: Sequence[float], data: np.ndarray) -> tuple[list, list]:
    """ Return predictions of normal and abnormal from svm model.

        :returns: Tuple of normal and abnormal intervals.
    """
    model = SVC()
    reshaped_training, reshaped_data = format_single_feature_data(train_data, data)
    predictions = estimator_prediction(
        model, reshaped_training, train_labels, reshaped_data)
    # 1.0 is normal, -1.0 is abnormal
    out = (prediction != 1.0 for prediction in predictions)
    return dense_to_time_intervals(time_vec, out)
