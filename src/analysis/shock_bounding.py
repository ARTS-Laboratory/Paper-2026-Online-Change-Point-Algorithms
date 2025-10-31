import math
from collections.abc import Sequence

import numpy as np
from more_itertools import windowed
from numpy.lib._stride_tricks_impl import sliding_window_view


def get_shock_start_and_end_times(time, data, safe_section, window_size: int, start_threshold_multiplier: float, end_cutoff_percentage: float):
    """ Estimate and return the start and end times for a signal with a shock event."""
    shock_start = get_shock_start(time, data, safe_section, window_size, start_threshold_multiplier)
    shock_end = get_shock_end(time, data, end_cutoff_percentage)
    return shock_start, shock_end

def get_shock_start(time, data, safe_section, window_size: int, threshold_multiplier: float = 2.0):
    """ """
    # shock start is when envelope goes above threshold
    envelope = get_envelope(data, window_size)
    # choose double the average envelope
    max_envelope = np.max(np.abs(safe_section))
    start_idx = np.where(envelope > max_envelope * threshold_multiplier)[0][0]
    start_time = time[start_idx + window_size]
    return start_time

def get_shock_start_end_by_envelope(
        time: np.ndarray, data: np.ndarray, safe_section: np.ndarray, window_size: int,
        threshold_multiplier: float = 2.0) -> tuple[float, float]:
    """ """
    # shock start is when envelope goes above threshold
    envelope = get_envelope_ndarray(data, window_size)
    # choose double the average envelope
    max_envelope = np.max(np.abs(safe_section))
    greater_than  = np.where(envelope > max_envelope * threshold_multiplier)[0]
    start_idx: int = greater_than[0]
    start_time: float = time[start_idx + window_size]
    stop_idx: int = greater_than[-1]
    stop_time: float = time[stop_idx + window_size]
    return start_time, stop_time


# def get_shock_start_index(data, safe_section, window_size: int, threshold_multiplier: float = 2.0):
#     """ """
#     # shock start is when envelope goes above threshold
#     envelope = get_envelope(data, window_size)
#     # choose double the average envelope
#     max_envelope = np.max(np.abs(safe_section))
#     start_idx = np.where(envelope > max_envelope * threshold_multiplier)[0][0]
#     return start_idx + window_size

def get_envelope(data: Sequence[float], window_size: int):
    """ Get envelope of data using given window size."""
    windows = windowed(data, window_size)
    return np.array([np.max(np.abs(window)) for window in windows])

def get_envelope_ndarray(data: np.ndarray, window_size: int):
    """ Get envelope of data using given window size."""
    windows = sliding_window_view(data, window_size)
    return np.max(np.abs(windows), axis=-1)

def get_shock_end(time, data, cutoff_percentage: float):
    """ Estimate and return stop time of shock for signal.

        If measurement does not go below cutoff, will return infinity.
    """
    abs_data = np.abs(data)
    peak = np.max(abs_data)
    # find last point where data is greater than or equal to x% of peak
    indices = np.where(abs_data >= cutoff_percentage * peak)
    last_high = indices[0][-1]
    # one over the index since this is the last
    stop_time = time[last_high + 1] if last_high + 1 < len(time) else math.inf
    return stop_time

# def get_shock_end_index(data, cutoff_percentage: float):
#     """ Estimate and return stop time of shock for signal.
#
#         If measurement does not go below cutoff, will return infinity.
#     """
#     abs_data = np.abs(data)
#     peak = np.max(abs_data)
#     # find last point where data is greater than or equal to x% of peak
#     indices = np.where(abs_data >= cutoff_percentage * peak)
#     # last index higher than or equal to cutoff
#     # one over the index since this is the last
#     return indices[0][-1] + 1
