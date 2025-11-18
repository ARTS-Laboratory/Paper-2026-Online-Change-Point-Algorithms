import itertools
from collections import namedtuple

import matplotlib.collections

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from change_point_algorithms.utils import metrics
from change_point_algorithms.utils.detection_arr_helpers import get_mean_array, convert_intervals_to_time, get_deviation_array


def interval_histogram(time, data, shock_intervals, non_shock_intervals, title=True):
    """ """
    shock_times = convert_intervals_to_time(time, shock_intervals)
    non_shock_times = convert_intervals_to_time(time, non_shock_intervals)
    fig, ax = plt.subplots(ncols=2)
    ax[0].hist([metrics.abs_mean(data[start:stop]) for start, stop in shock_times])
    ax[1].hist([metrics.abs_mean(data[start:stop]) for start, stop in non_shock_times])
    return fig


def raw_histogram(time, data, shock_intervals, non_shock_intervals, title=True):
    """ """
    n_bins = 50
    shock_times = convert_intervals_to_time(time, shock_intervals)
    non_shock_times = convert_intervals_to_time(time, non_shock_intervals)
    # These flatten the list of lists into one big list
    # shock_vals = [point for start, stop in shock_times for point in data[start:stop]]
    # non_shock_vals = [point for start, stop in non_shock_times for point in data[start:stop]]
    fig, ax = plt.subplots(ncols=2, sharey=True)
    ax[0].hist(np.array(list(itertools.chain([data[start:stop] for start, stop in shock_times]))), bins=n_bins)
    ax[1].hist(np.array(list(itertools.chain([data[start:stop] for start, stop in non_shock_times]))), bins=n_bins)
    return fig


def plot_shock_helper(ax, interval, color, alpha):
    """ Helper function to plot shading rectangles as collection.

        :param plt.Axes ax: Axes to plot rectangles.
        :param List[tuple[float, float]] interval: Iterable of tuples containing start and stop times.
        :param str color: Color of rectangles.
        :param float alpha: Transparency of rectangles.
    """
    ax.add_collection(matplotlib.collections.PatchCollection(
        (Rectangle((start, 0), stop - start, 1.0) for start, stop in interval),
        alpha=alpha, facecolor=color, transform=ax.get_xaxis_transform()))


def plot_shock(time, data, shock_intervals, non_shock_intervals, to_ms=False):
    # fig = plt.figure(figsize=(6.5, 2), layout='compressed')
    # ax = plt.gca()
    fig, ax = plt.subplots(figsize=(6.5, 2), layout='compressed')
    safe_color = 'blue'
    unsafe_color = 'red'
    shade_alpha = 0.3
    y_min = -200
    y_max = 200
    plot_shock_v1(
        ax, time, data, shock_intervals, non_shock_intervals, y_min, y_max,
        to_ms=True, legend=True)
    # plt.tight_layout()
    return fig

def plot_shock_v1(ax: plt.Axes, time, data, shock_intervals, non_shock_intervals, y_min, y_max, **kwargs):
    """ """
    # Keyword stuff
    to_ms = kwargs.get('to_ms', False)
    safe_color = kwargs.get('safe_color', 'blue')
    unsafe_color = kwargs.get('unsafe_color', 'red')
    shade_alpha = kwargs.get('shade_alpha', 0.3)
    legend = kwargs.get('legend', True)
    # Start plotting
    if to_ms:
        scalar = 1_000
        ms_time = time * scalar
        # Plot acceleration
        ax.plot(ms_time, data, color='black')
        ax.set_xlim((ms_time[0], ms_time[-1]))
        new_shocks = ((start * scalar, stop * scalar) for start, stop in shock_intervals)
        new_non_shocks = ((start * scalar, stop * scalar) for start, stop in non_shock_intervals)
        plot_shock_helper(ax=ax, interval=new_shocks, color=unsafe_color, alpha=shade_alpha)
        plot_shock_helper(ax=ax, interval=new_non_shocks, color=safe_color, alpha=shade_alpha)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('acceleration (m/s\u00b2)')
    else:
        # Plot acceleration
        ax.plot(time, data, color='black')
        ax.set_xlim((time[0], time[-1]))
        # Shade regions
        plot_shock_helper(ax=ax, interval=shock_intervals, color=unsafe_color, alpha=shade_alpha)
        plot_shock_helper(ax=ax, interval=non_shock_intervals, color=safe_color, alpha=shade_alpha)
        # Setting plot details
        ax.set_xlabel('time (s)')
        ax.set_ylabel('acceleration (m/s\u00b2)')
    ax.set_ylim((y_min, y_max))
    if legend:
        safe_rect = Rectangle((0, 0), 1, 1, facecolor=safe_color, alpha=shade_alpha)
        unsafe_rect = Rectangle((0, 0), 1, 1, facecolor=unsafe_color, alpha=shade_alpha)
        ax.legend((safe_rect, unsafe_rect), ('normal region', 'shock region'), bbox_to_anchor=(1, 1), loc='lower right', ncol=2)
    return ax

def plot_shock_and_zoomed_start(
        ax: plt.Axes, time: np.ndarray, data: np.ndarray, shock_intervals,
        non_shock_intervals, x_left, x_right, to_ms=False, **kwargs):
    """ """
    safe_color = 'blue'
    unsafe_color = 'red'
    shade_alpha = 0.3
    y_min = kwargs.get('y_min', -200)
    y_max = kwargs.get('y_max', 200)
    shock_start: float | None = kwargs.get('shock_start', None)
    zoomed_y_min = kwargs.get('zoomed_y_min', -10)
    zoomed_y_max = kwargs.get('zoomed_y_max', 10)
    plot_shock_v1(
        ax, time, data, shock_intervals, non_shock_intervals, y_min, y_max,
        to_ms=to_ms, legend=True)
    if to_ms:
        data_region = DataRegion(x_left * 1_000, x_right * 1_000, y_min, y_max)
    else:
        data_region = DataRegion(x_left, x_right, y_min, y_max)
    inset_location = Position(0.1, 0.45)
    _, axins = zoom_in_for_detection(
        ax, data_region, inset_location, 0.2, 0.5)
    if shock_start is not None:
        start = shock_start if not to_ms else shock_start * 1_000
        # print(f'Draw a line: at x={start}')
        ax.axvline(x=start, linestyle='--')
        axins.axvline(x=start, linestyle='--')
        # axins.axvline(x=141.25)
    plot_shock_v1(
        axins, time, data, shock_intervals, non_shock_intervals,
        zoomed_y_min, zoomed_y_max, to_ms=to_ms, legend=False)
    axins.set_xlim((data_region.left, data_region.right))
    axins.set_xlabel('')
    axins.set_ylabel('')
    # if to_ms:
    #     axins.set_xlim((x_left * 1_000, x_right * 1_000))
    # else:
    #     axins.set_xlim((x_left, x_right))
    return ax

DataRegion = namedtuple('DataRegion', ['left', 'right', 'bottom', 'top'])
Position = namedtuple('Position', ['x', 'y'])

def zoom_in_for_detection(
        ax: plt.Axes, data_region: DataRegion, inset_location: Position,
        inset_width: float, inset_height: float):
    """ """
    axins = ax.inset_axes(
        (inset_location.x, inset_location.y, inset_width, inset_height),
        xlim=(data_region.left, data_region.right),
        ylim=(data_region.bottom, data_region.top))
    ax.indicate_inset_zoom(axins, edgecolor='black')
    return ax, axins

def plot_shock_w_mean_std(time, data, shock_intervals, non_shock_intervals):
    means = get_mean_array(time, data, shock_intervals, non_shock_intervals)
    devs = get_deviation_array(time, data, shock_intervals, non_shock_intervals)
    fig = plt.figure()
    # Plot acceleration
    plt.plot(time, data, color='black')
    plt.plot(time, means, '--', color='black')
    plt.plot(time, means + devs, ':', color='black')
    plt.plot(time, means - devs, ':', color='black')
    # Shade regions
    ax = plt.gca()
    plot_shock_helper(ax, shock_intervals, 'red', 0.3)
    plot_shock_helper(ax, non_shock_intervals, 'blue', 0.3)
    # Setting plot details
    plt.xlim((time[0], time[-1]))
    plt.xlabel('time (s)')
    plt.ylabel('acceleration (m/s\u00b2)')
    plt.title('Forced Vibration And Shock (Blue=Shock, Red=Non-shock)')
    plt.tight_layout()
    return fig

def plot_shock_and_zoomed_for_paper(time, data, shock_intervals, non_shock_intervals, x_left, x_right, to_ms=False, **kwargs):
    """ """
    fig, ax = plt.subplots(figsize=(6.5, 2), layout='compressed')
    y_min = -200
    y_max = 200
    plot_shock_and_zoomed_start(
        ax, time, data, shock_intervals, non_shock_intervals, x_left, x_right,
        to_ms, y_min=y_min, y_max=y_max, zoomed_y_min=-20, zoomed_y_max=20, **kwargs)
    return fig
