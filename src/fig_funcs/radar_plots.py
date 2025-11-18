import matplotlib as mpl
import numpy as np
import pandas as pd
# import plotly.graph_objects as go
from matplotlib import pyplot as plt

from fig_funcs._radar_chart import radar_factory
from utils.plotly_formatting import update_font


# def plot_radar_single_matplotlib(ax, data, ax_labels, label, fill):
#     """ """

def metric_scores_matplotlib_data(ax, theta, values, **kwargs):
    """ """
    colors = kwargs.get('colors', mpl.color_sequences['Set1'])
    colors = kwargs['colors'] if 'colors' in kwargs else mpl.color_sequences['Set1']
    for d, color in zip(values, colors):
        ax.plot(theta, d, color=color)
        if fill:
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_no_legend_')

def plot_metric_scores_matplotlib(ax, theta, values, ax_labels, labels, fill=True):
    """ Plot metric scores """
    colors = mpl.color_sequences['Set1']
    for d, color in zip(values, colors):
        ax.plot(theta, d, color=color)
        if fill:
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
    # Aesthetics
    ax.set_varlabels(ax_labels)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_rlabel_angle(0)
    val = 0.0  # 45.0
    real = 2.5  # 0.5
    angle = np.deg2rad(val)
    # ax.legend(labels, loc='lower left',
    #           bbox_to_anchor=(0.5 + real*np.cos(angle)/2, 0.5 + real*np.sin(angle)/2))
    # ax.legend(labels, loc='upper right', bbox_to_anchor=(1, 1))
    return ax

# def plot_metric_bars(ax, values, )

# def plot_radar_single(fig: go.Figure, values, theta, name, fill):
#     """ Add radar plot for a single metric table to figure."""
#     fig.add_trace(
#         go.Scatterpolar(r=values, theta=theta, fill = fill, name=name))
#     return fig


# def plot_metric_scores_plotly(fig: go.Figure, data_collection, axis_names, labels, fill=None):
#     """ """
#     fill_type = fill if fill is not None else 'toself'
#     # append beginning to end to close chart
#     # make chart
#     update_font(fig)
#     for (data, label) in zip(data_collection, labels):
#         plot_radar_single(fig, data, axis_names, label, fill_type)
#     return fig

def plot_metric_scores_for_paper(df: pd.DataFrame):
    """ """
    _metric_names = (
        'accuracy', 'precision', 'recall', 'f1 score', 'earliest correct',
        'delay')
    # metric_pretty_names = (
    #     'Accuracy', 'Precision', 'Recall', 'F1 score',
    #     'Earliest Correct (ms)', 'Detection Delay (ms)')
    metric_pretty_names = (
        'Accuracy', 'Precision', 'Recall', 'F1 score',
        'Detection Delay (ms)')
    # todo confirm metrics exist in table
    # todo confirm algorithm names match
    alg_pretty_names = ('BOCPD', 'EM', 'GM', 'CUSUM')
    skip_idx = 2
    df_copy = df.copy(deep=True)
    scalar = 1_000
    # df_copy['earliest correct'] *= scalar
    # df_copy['delay'] *= scalar
    print(f'detection delay \n{df_copy["delay"] * scalar}')
    data = df_copy[[
        'accuracy', 'precision', 'recall',
        'f1 score', 'delay']].itertuples(index=False)
    data_collection = tuple(item for item in data)
    # Matplotlib version
    size = len(metric_pretty_names)
    theta = radar_factory(size, frame='circle')
    fig, ax = plt.subplots(figsize=(5, 2), layout='constrained', subplot_kw=dict(projection='radar'))
    plot_metric_scores_matplotlib(ax, theta, data_collection, metric_pretty_names, alg_pretty_names, fill=True)
    return fig
    # Plotly version
    fig = go.Figure()
    plot_metric_scores_plotly(fig, data_collection, metric_pretty_names, alg_pretty_names)
    return fig

