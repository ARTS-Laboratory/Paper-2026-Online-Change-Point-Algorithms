import itertools
import os
from functools import partial

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

from analysis.shock_bounding import get_shock_start_and_end_times, get_shock_start_end_by_envelope
from evals.ground_truth import get_ground_truth_from_file, generate_ground_truth
from evals.scores import get_select_scores
from fig_funcs.detection_plots import plot_shock_and_zoomed_for_paper
from parsing.data_parsing import load_data_from_config
from utils.path_validation import confirm_dir_or_consult
from utils.read_data import load_signals
from utils.toml_utils import load_toml


def parse_eval_online(config_file):
    """ """
    config_table = load_toml(config_file)
    version = config_table['version']
    if version == "1":
        ...
        # config_table = load_toml(config_file)
        # file_path = Path(config_table['file-path'])
        # time, data = load_signals(file_path)
        # # time, data, results, ground
        # plot_detection_online_models(time, data, results)
        # df = evaluate_online_models(time, results, ground)
    elif version == "2":
        parse_eval_online_for_v2(config_table)
    else:
        pass

def parse_eval_online_for_v2(config_table):
    """
    :param config_table: TOML table loaded from config file.
    :return:
    """
    save_config = config_table['saves']
    eval_config = config_table['eval']
    # Load data
    eval_data_config = eval_config['data']
    df = parse_eval_data_config(eval_data_config, save_config)
    print(df.head())
    ground = parse_eval_ground(eval_config['ground'], config_table['data'])
    print(ground)
    # Load data
    data_config = config_table['data']
    time, data = load_data_from_config(data_config)
    # Evaluate
    metrics_config = eval_config['metrics']
    score_df = parse_eval_metrics_config(metrics_config, time, ground, df)
    # Save
    eval_save_config = eval_config['saving']
    root = save_config['save-root'] if eval_save_config['is-subdir'] else None
    parse_eval_save_config(eval_save_config, score_df, root)
    # Plotting detection
    plot_detection(time, data, df, save_dir=root)

def plot_detection(time, data, df, save_dir=None):
    """ """
    root = '' if save_dir is None else save_dir
    interval_dict = df_to_intervals(df, time[-1])
    start, stop = get_shock_start_and_end_times(
        time, data, data[:100_000], 1_000, 2, 0.95
    )
    print(f'Shock started {start:.6} seconds in and ended {stop:.6} seconds in.')
    # x_left, x_right = 141.0e-3, 141.5e-3
    x_left, x_right = 141.0e-3, 142.0e-3
    for name, (shocks, nonshocks) in interval_dict.items():
        print(f'Detection plot for {name}')
        detection_fig = plot_shock_and_zoomed_for_paper(
            time, data, shocks, nonshocks, x_left, x_right, to_ms=True, shock_start=start
        )
        # detection_fig.gca().axvline(x=x_left * 1_000, linestyle='--')
        detection_fig.savefig(Path(root, f'signal-1-detection-{name}.png'), dpi=300)
        # detection_fig.show()
        plt.close(detection_fig)

def parse_eval_metrics_config(metrics_config, time, ground, df):
    """ """
    metric_names = metrics_config['scores']
    predictions = df.select('name', 'prediction').group_by('name')
    metric_fn = partial(get_select_scores, time, ground, metrics=metric_names)
    scores = [(name[0],) + tuple(metric_fn(vals.select('prediction').to_numpy(order='c').flatten()).values()) for name, vals in predictions]
    score_df = pl.DataFrame(scores, schema=['name']+metric_names, orient='row')
    print(score_df)
    formatted_score_df = score_df.select(
        'name',
        pl.col('accuracy').round(3),
        pl.col('precision').round(3),
        pl.col('recall').round(3),
        pl.col('f1-score').round(3),
        pl.col('delay').alias('delay (s)'),
        (pl.col('delay') * 1_000).alias('delay (ms)'),
    )
    print(formatted_score_df)
    return score_df

def parse_eval_data_config(eval_data_config, save_config) -> pl.DataFrame:
    """ """
    match eval_data_config:
        case {'what': 'csv', 'where': {'is-subdir': True, 'dir': root, 'filename': name}}:
            super_root = save_config['save-root']
            file_path = Path(super_root, root, name)
        case {'what': 'csv', 'where': {'is-subdir': False, 'dir': root, 'filename': name}}:
            file_path = Path(root, name)
        case x:
            raise NotImplementedError(f'No loading implementation for type {x}')
    return pl.scan_csv(file_path.with_suffix('.csv')).collect()
    # match eval_data_config['what']:
    #     case 'csv':
    #         where = eval_data_config['where']
    #         if where['is-subdir']:
    #             file_path = Path(
    #                 save_config['save-root'], where['dir'], where['filename'])
    #         else:
    #             file_path = Path(where['dir'], where['filename'])
    #         return pl.scan_csv(file_path.with_suffix('.csv')).collect()
    #     case x:
    #         raise NotImplementedError(f'No loading implementation for type {x}')
    # return

def parse_eval_ground(eval_ground, data_config):
    """ Parse 'ground' table of eval config."""
    # Get ground truth
    ## Optionally: save ground truth
    match eval_ground:
        case {'what': 'save', 'save': {'name': name, 'dir': folder}}:
            eval_path = Path(folder, name)
            return get_ground_truth_from_file(eval_path)
        case {
            'what': 'generate', 'generate': {
                'alg': alg, 'name': _gen_name, 'extras': extras,
                'saving': gen_save}
        }:
            # Load data
            time, data = load_data_from_config(data_config)
            # Generate ground
            ground = generate_ground_truth(data, alg, extras)
            # now check if you want to save this for later.
            if gen_save['save']:
                match gen_save['what']:
                    case 'npy' | 'numpy':
                        dir = gen_save['where']['dir']
                        name = gen_save['where']['save-name']
                        np.save(Path(dir, name), ground)
                    case x:
                        raise NotImplementedError(f'No implementation for saving generated ground truth for {x}')
            return ground
        case {'what': x}:
            raise ValueError(f'No option "{x}" for ground truth retrieval.')
    # match eval_ground['what']:
    #     case 'save':
    #         dir = eval_ground['save']['dir']
    #         name = eval_ground['save']['name']
    #         eval_path = Path(dir, name)
    #         return get_ground_truth_from_file(eval_path)
    #     case 'generate':
    #         gen_config = eval_ground['generate']
    #         alg = gen_config['alg']
    #         _gen_name = gen_config['name']
    #         extras = gen_config.get('extras')
    #         # Load data
    #         time, data = load_data_from_config(data_config)
    #         # Generate ground
    #         ground = generate_ground_truth(data, alg, extras)
    #         # now check if you want to save this for later.
    #         gen_save = gen_config['saving']
    #         if gen_save['save']:
    #             match gen_save['what']:
    #                 case 'npy' | 'numpy':
    #                     dir = gen_save['where']['dir']
    #                     name = gen_save['where']['save-name']
    #                     np.save(Path(dir, name), ground)
    #                 case x:
    #                     raise NotImplementedError(f'No implementation for saving generated ground truth for {x}')
    #         return ground
    #     case x:
    #         raise ValueError(f'No option "{x}" for ground truth retrieval.')

def df_to_intervals(df: pl.DataFrame, last_time: float) -> dict[str, tuple]:
    """ Convert dataframe with reported change point times into dictionary of interval pairs."""
    interval_dict = dict()
    # interval_df = df.filter(is_change_point=True).select('name', 'time')
    interval_dfs = df.filter(is_change_point=True).select('name', 'time').partition_by('name')
    for interval_df in interval_dfs:
        name = interval_df.get_column('name').item(0)
        shock = False
        shocks = list()
        nonshocks = list()
        times = interval_df.get_column('time')
        for pair in itertools.pairwise(times):
            if shock:
                shocks.append(pair)
            else:
                nonshocks.append(pair)
            shock = not shock
        if (end_start := times.last()) <= last_time:
            last_region = (end_start, last_time)
            if shock:
                shocks.append(last_region)
            else:
                nonshocks.append(last_region)
        interval_dict[name] = (shocks, nonshocks)
    return interval_dict

def parse_eval_save_config(
        eval_save_config, score_df: pl.DataFrame,
        save_root: os.PathLike | None = None):
    """ Parse 'saving' table of eval config."""
    root = save_root if save_root is not None else ''
    if eval_save_config['save']:
        folder = eval_save_config['where']
        save_name = eval_save_config['save-name']
        save_path = Path(root, folder, save_name)
        match eval_save_config['how']:
            case 'csv': #str(x) if x == 'csv':
                score_df.write_csv(save_path.with_suffix('.csv'))
            case 'tex-table': # str(x) if x == 'tex-table':
                raise NotImplementedError
            case None:
                raise TypeError('[eval.saving] table is missing "how" key.')

# def something_else(config_file):
#     ## Process online models
#     df = process_online_models(time, data, algs, ground)
#     # results = run_online_models(time, data, algs)
#     # plot_detection_online_models(time, data, results)
#     # df = evaluate_online_models(time, results, ground)
#     ## Process offline anomaly models
#     anom_save_dir = config_table['save-root']
#     confirm_dir_or_consult(Path(anom_save_dir))
#     # anom_save_dir = Path(os.curdir, 'figures', '2025-04-07', 'signal-1-filtered')
#     anom_save_name = Path('isolation_forest_fig')
#     data_size = len(data)
#     half_point = data_size//2
#     safe = data[:half_point]
#     unsafe = data[half_point:]
#     anom_results = run_offline_anomaly_models(time, safe, unsafe, data, (AnomalyType.ISO_FOREST,))
#     plot_detection_anomaly_models(time, data, anom_results, anom_save_dir, anom_save_name)
#     ## Write data frame for LaTeX
#     save_root = config_table['save-root']
#     metric_table_config = config_table['metric-table']
#     metric_root = metric_table_config['save-root']
#     save_name = metric_table_config['save-name']
#     save_folder = Path(save_root, metric_root)
#     confirm_dir_or_consult(save_folder)
#     df_reduced = df.loc[:, ['accuracy', 'precision', 'recall', 'f1 score', 'delay']]
#     tex_table = format_frame_for_latex(df)
#     tex_table_reduced = tex_table.loc[:, ['accuracy', 'precision', 'recall', 'f1 score', 'detection delay (ms)']]
#     # save_folder = Path(os.curdir, 'figures', '2025-02-11', 'tables')
#     # save_name = config_table['metric-table']['save-name']
#     write_frame_to_latex(tex_table_reduced, save_name, save_folder)
#     # Make radar chart
#     # metric_radar_fig = plot_metric_scores_for_paper(df)
#     # metric_fig_2 = plot_metrics_as_bar(df_reduced)
#     metric_fig_3 = plot_metrics_as_radar_bar_for_paper(df_reduced)
#     plt.savefig(Path(save_root, 'metric-radar-chart.png'), dpi=350)
#     # metric_radar_fig.write_image(Path(save_root, 'metric_radar_chart.png'))
#     # metric_radar_fig.write_image(Path(save_root, 'metric_radar_chart.pdf'))
