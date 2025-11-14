from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from evals.ground_truth import make_ground_truth
from fig_funcs.detection_plots import plot_shock
from offline_detection import binary_segmentation
from plot_makers.signal_plot_makers import make_signal_plots, make_spectrogram_plots
from utils.write_data import save_path


def plot_signals(time, data, show=False, save_root=None):
    """ """
    save_dir = save_path(save_root)
    make_signal_plots(time, data, save_root=save_dir)
    make_spectrogram_plots(time, data, save_root=save_dir)
    if show:
        plt.show()

def plot_offline_detections(time, data, ground=None, save_root=None):
    """ Plot figures for shock detection via offline detection algorithms."""
    save_dir = save_path(save_root)
    num_bkps = 2
    bkps = binary_segmentation.get_breaks(np.abs(data), num_bkps, model_type='rank')
    binary_segmentation.plot_breaks(data, bkps)
    # plt.savefig('./figures/offline_rank_bin-seg.jpg', dpi=350)
    # bkps = bottom_up.get_breaks(np.abs(data), num_bkps)
    # bottom_up.plot_breaks(data, bkps)
    # bkps = dynamic_programming.get_breaks(np.abs(data), num_bkps)
    # rupture_changepoint_plots.plot_breaks(data, bkps, show=True)
    # Make ground truth plot
    ground_shocks_idx, ground_nonshocks_idx = make_ground_truth(data)
    ground_shocks = [(time[start], time[stop - 1]) for start, stop in ground_shocks_idx]
    ground_nonshocks = [(time[start], time[stop - 1]) for start, stop in ground_nonshocks_idx]
    print('Shock event start and stop times')
    for start, stop in ground_shocks:
        print(f'Shock event start: {start}, shock event stop: {stop}')
    ground_truth_fig = plot_shock(time, data, ground_shocks, ground_nonshocks)
    # plt.savefig(Path(save_dir, 'ground_truth_fig.pdf'))
    # plt.savefig(Path(save_dir, 'ground_truth_fig.png'), dpi=350)
    plt.close(ground_truth_fig)