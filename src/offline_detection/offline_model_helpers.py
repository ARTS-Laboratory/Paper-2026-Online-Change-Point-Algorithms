from collections.abc import Sequence
from typing import Iterable

import numpy as np


def format_single_feature_data(training_data: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Expand dimension for data with single feature."""
    reshaped_training = np.expand_dims(training_data, axis=1)
    reshaped_data = np.expand_dims(data, axis=1)
    return reshaped_training, reshaped_data


def estimator_prediction(estimator, fitting_data, fitting_labels, data: np.ndarray):
    """ """
    estimator.fit(fitting_data, fitting_labels)
    predictions = estimator.predict(data)
    return predictions


def dense_to_time_intervals(time_vec: Sequence[float], predictions: Iterable[bool]):
    """ Convert dense array of predictions to time interval"""
    shocks = list()
    steady = list()
    begin = 0
    shock = False
    for idx, is_change in enumerate(predictions):
        if is_change and not shock:
            steady.append((time_vec[begin], time_vec[idx]))
            shock = True
            begin = idx
        elif not is_change and shock:
            shocks.append((time_vec[begin], time_vec[idx]))
            shock = False
            begin = idx
    if shock:
        shocks.append((time_vec[begin], time_vec[-1]))
    else:
        steady.append((time_vec[begin], time_vec[-1]))
    return shocks, steady
