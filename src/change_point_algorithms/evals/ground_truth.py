import os
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from analysis.shock_bounding import get_shock_start_end_indices_by_envelope
from offline_detection import binary_segmentation
from utils.detection_arr_helpers import convert_interval_indices_to_full_arr


def make_ground_truth(data):
    """ Calculates ground truth and returns indices for intervals"""
    shocks, nonshocks = list(), list()
    num_bkps = 2
    bkps = binary_segmentation.get_breaks(np.abs(data), num_bkps, model_type='rank')
    begin = 0
    shocked = False
    print(f'number of breakpoints: {len(bkps)}')
    for bkp in bkps[:-1]:
        entry = (begin, bkp + 1)
        if shocked:
            shocks.append(entry)
        else:
            nonshocks.append(entry)
        shocked = not shocked
        begin = bkp
    entry = (begin, bkps[-1])
    if shocked:
        shocks.append(entry)
    else:
        nonshocks.append(entry)
    return shocks, nonshocks

class EnvelopeGround:
    """ """
    def __init__(self, data, safe_section, window_size, threshold_multiplier):
        self.data = data
        self.safe = safe_section
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier

    @classmethod
    def from_data_and_dict(cls, data, ground_dict):
        """ """
        safe_start: int = ground_dict['safe_start']
        safe_end: int = ground_dict['safe_end']
        safe = data[safe_start:safe_end]
        window_size = ground_dict['window_size']
        threshold_multiplier = ground_dict['threshold_multiplier']
        return cls(data, safe, window_size, threshold_multiplier)

    def generate(self):
        start_idx, stop_idx = get_shock_start_end_indices_by_envelope(
            self.data, self.safe, self.window_size, self.threshold_multiplier)
        data_len = len(self.data)
        shocks = [(start_idx, stop_idx)]
        nonshocks = [(0, start_idx - 1), (stop_idx, data_len - 1)]
        return convert_interval_indices_to_full_arr(shocks, nonshocks, data_len)


def generate_ground_truth(data, alg, alg_context: Mapping | None = None):
    match alg:
        case '' | 'binary segmentation':
            (true_shocks, true_nonshocks) = make_ground_truth(data)
            return convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(data))
        case 'envelope':
            ground_alg = EnvelopeGround.from_data_and_dict(data, alg_context)
            return ground_alg.generate()
            # start_idx = get_shock_start_index(data, alg_context.time)
        case str(x):
            raise ValueError(f'"{x}" is not a recognized algorithm.')
        case None:
            raise ValueError('algorithm name not found')
        case _:
            raise ValueError(f'Unknown algorithm {alg}')

def get_ground_truth_from_file(file: os.PathLike):
    """ Load ground truth data from file."""
    file_path = Path(file)
    match file_path.suffix:
        case '.npy':
            return np.load(file)
        case _:
            raise NotImplementedError(f'No implementation for file type {file_path.suffix}')
