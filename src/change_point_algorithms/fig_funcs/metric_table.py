import string
from typing import Optional

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fig_funcs._radar_chart import radar_factory
from fig_funcs.radar_plots import plot_metric_scores_matplotlib
# def get_default_colors()

def bar_plot(ax: plt.Axes, values, bar_width: float, colors: Optional[list[str]] = None, labels: Optional[list[str]] = None, annotate=False):
    """ """
    num_bars = len(values)
    if labels is None:
        bar_container = ax.bar(range(num_bars), values, bar_width)
    else:
        bar_container = ax.bar(labels, values, bar_width, color=colors, label=labels)
    if annotate:
        ax.bar_label(bar_container, fmt='{:.3f}')
    return ax


def grouped_bar_plot(ax: plt.Axes, g_values, bar_width, labels=None, annotate=False):
    """ """
    if isinstance(g_values, dict):
        pass
    elif isinstance(g_values, np.ndarray):
        # length = g_values.shape[0]
        # group_size = g_values.shape[1]
        # group_offset = range(group_size)
        for idx, group in enumerate(g_values):
            ax.bar(idx, group, bar_width)
    elif isinstance(g_values, pd.DataFrame):
        # iterate over dataframe by rows
        # num_entries = g_values.shape[0]
        num_bars = g_values.shape[1]
        new_bar_width = 1 / (g_values.shape[0] + 1)
        offset = -(new_bar_width * g_values.shape[0] / 2) + (new_bar_width / 2)
        category_positions = np.array(range(num_bars))
        colors = mpl.color_sequences['Set1']
        for row, color in zip(g_values.itertuples(), colors):
            idx = row[0]
            entries = row[1:]
            bar_container = ax.bar(
                category_positions
                + offset
                + (idx * new_bar_width),
                entries, new_bar_width, color=color)
            if annotate:
                # ax.bar_label(bar_container, fmt='{:,.3f}')
                ax.bar_label(bar_container, fmt='{:.3f}')
    return ax


def metric_bar_plot(ax: plt.Axes, values: np.ndarray, labels, unit: str):
    """ Plot bar plots for metric table"""
    ax.bar(labels, values)


def plot_metrics_as_bar(df: pd.DataFrame) -> plt.Figure:
    """ """
    _metric_names = (
        'accuracy', 'precision', 'recall', 'f1 score', 'earliest correct',
        'delay')
    metric_pretty_names = (
        'accuracy', 'precision', 'recall', 'f1 score')
    alg_pretty_names = ('BOCPD', 'EM', 'GM', 'CUSUM')
    skip_idx = 2
    df_copy = df.copy(deep=True)
    scalar = 1_000
    data = df_copy[[
        'accuracy', 'precision', 'recall',
        'f1 score']].itertuples(index=False)
    data_collection = tuple(item for item in data)
    size = len(metric_pretty_names)
    theta = radar_factory(size, frame='circle')
    # size = len(metric_pretty_names)
    # fig, ax = plt.subplots(ncols=2, figsize=(5, 2), layout='constrained')
    fig: plt.Figure = plt.figure(figsize=(6.5, 3), layout='constrained')
    radar_ax: plt.Axes = fig.add_subplot(1, 2, 1, projection='radar')
    # Plot radar chart
    plot_metric_scores_matplotlib(
        radar_ax, theta, data_collection, metric_pretty_names, alg_pretty_names,
        fill=True)
    # Plot bar chart
    bar_ax: plt.Axes = fig.add_subplot(1, 2, 2)
    data = df_copy.loc[:, ['delay']] * scalar
    bar_width = 0.5
    grouped_bar_plot(bar_ax, data, bar_width, annotate=True)
    metric_pretty_names = ['delay']
    # todo change magic number to rounded up integer
    bar_ax.set_ylim(0.0, 6.0)
    bar_ax.set_xticks(range(len(metric_pretty_names)), metric_pretty_names)
    bar_ax.set_ylabel('time elapsed (ms)')
    # # now we can make the legend
    # bar_ax.legend(labels=alg_pretty_names, loc='upper right', bbox_to_anchor=)
    return fig

# def plot_metrics_bar(ax: plt.Axes)

def plot_metrics_as_radar_bar_for_paper(df: pd.DataFrame) -> plt.Figure:
    """ Call a version of metric plotting for paper."""
    return plot_metrics_as_radar_bar_for_paper_2(df)

def plot_metrics_as_radar_bar_for_paper_1(df: pd.DataFrame) -> plt.Figure:
    """ Plot side-by-side radar and bar plot."""
    _metric_names = (
        'accuracy', 'precision', 'recall', 'f1 score', 'earliest correct',
        'delay')
    metric_pretty_names = (
        'accuracy', 'precision', 'recall', 'f1 score')
    alg_pretty_names = ('BOCPD', 'EM', 'GM', 'CUSUM')
    df_copy = df.copy(deep=True)
    scalar = 1_000
    data = df_copy[[
        'accuracy', 'precision', 'recall',
        'f1 score']].itertuples(index=False)
    data_collection = tuple(item for item in data)
    size = len(metric_pretty_names)
    theta = radar_factory(size, frame='circle')
    fig: plt.Figure = plt.figure(figsize=(6.5, 3), layout='constrained')
    radar_ax: plt.Axes = fig.add_subplot(1, 2, 1, projection='radar')
    # Plot radar chart
    plot_metric_scores_matplotlib(
        radar_ax, theta, data_collection, metric_pretty_names, alg_pretty_names,
        fill=True)
    # this line removes the legend created in the above function
    radar_ax.get_legend().remove()
    # Plot bar chart
    bar_ax: plt.Axes = fig.add_subplot(1, 2, 2)
    data = df_copy.loc[:, 'delay'].array * scalar
    bar_width = 0.5
    num_algs = len(data)
    colors = mpl.color_sequences['Set1'][:num_algs]
    bar_plot(bar_ax, data, bar_width, colors=colors, labels=list(alg_pretty_names), annotate=True)
    # todo change magic number to rounded up integer
    bar_ax.set_ylim(0.0, 6.0)
    bar_ax.set_ylabel('time elapsed (ms)')
    bar_ax.tick_params(axis='x', labelrotation=30.0)
    # bar_ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncols=1)
    bar_ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncols=num_algs // 2)
    # Add labels
    add_labels_to_subfigures([radar_ax, bar_ax])
    return fig

def plot_metrics_as_radar_bar_for_paper_2(df: pd.DataFrame) -> plt.Figure:
    """ Plot side-by-side radar and bar plot."""
    _metric_names = (
        'accuracy', 'precision', 'recall', 'f1 score', 'earliest correct',
        'delay')
    metric_pretty_names = (
        'accuracy', 'precision', 'recall', 'f1 score')
    alg_pretty_names = ('BOCPD', 'EM', 'GM', 'CUSUM')
    df_copy = df.copy(deep=True)
    scalar = 1_000
    data = df_copy[[
        'accuracy', 'precision', 'recall',
        'f1 score']].itertuples(index=False)
    data_collection = tuple(item for item in data)
    size = len(metric_pretty_names)
    theta = radar_factory(size, frame='circle')
    fig: plt.Figure = plt.figure(figsize=(6.5, 3.0), layout='constrained')
    radar_fig: plt.Figure
    bar_fig: plt.Figure
    radar_fig, bar_fig = fig.subfigures(1, 2)
    # Radar Chart
    # radar_ax: plt.Axes = radar_fig.subplots(1, 1)
    radar_ax: plt.Axes = radar_fig.add_subplot(1, 1, 1, projection='radar')
    ## Plot radar chart
    plot_metric_scores_matplotlib(
        radar_ax, theta, data_collection, metric_pretty_names, alg_pretty_names,
        fill=True)
    # this line removes the legend created in the above function
    # radar_ax.get_legend().remove()
    # Bar Chart
    bar_ax: plt.Axes = bar_fig.subplots(1, 1)
    data = df_copy.loc[:, 'delay'].array * scalar
    bar_width = 0.5
    num_algs = len(data)
    colors = mpl.color_sequences['Set1'][:num_algs]
    bar_plot(bar_ax, data, bar_width, colors=colors, labels=list(alg_pretty_names), annotate=True)
    # todo change magic number to rounded up integer
    bar_ax.set_ylim(0.0, 6.0)
    bar_ax.set_ylabel('time elapsed (ms)')
    bar_ax.tick_params(axis='x', labelrotation=30.0)
    # bar_ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncols=1)
    bar_ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncols=num_algs // 2)
    # Add labels
    add_labels_to_subfigures([radar_ax, bar_ax])
    return fig


def add_labels_to_subfigures(axs: list[plt.Axes]):
    """ """
    # assert number of ax subfigs is under 26
    assert len(axs) < 27
    for letter, ax in zip(string.ascii_lowercase, axs):
        label = f'({letter})'
        # ax.annotate(label, xy=(0.5, 0), xycoords='subfigure fraction',
        #             # xytext=(0.0, -1.0), textcoords='offset fontsize',
        #             # bbox=dict(pad=2.0),
        #             )
        ax.annotate(label, xy=(0.5, 0.025), xycoords='subfigure fraction',
                    # xytext=(0.0, -1.0), textcoords='offset fontsize',
                    # bbox=dict(pad=2.0),
                    horizontalalignment='center',
                    )
