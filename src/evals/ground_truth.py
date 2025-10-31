import os
from pathlib import Path

import numpy as np

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

def generate_ground_truth(data, alg=None, **kwargs):
    if alg == '':
        (true_shocks, true_nonshocks) = make_ground_truth(data)
        ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(data))
    elif alg == 'shock bounding':
        start_idx = get_shock_start_index(data, )
    return ground

def get_ground_truth_from_file(file: os.PathLike):
    """ Load ground truth data from file."""
    file_path = Path(file)
    match file_path.suffix:
        case '.npy':
            return np.load(file)
        case _:
            raise NotImplementedError(f'No implementation for file type {file_path.suffix}')
