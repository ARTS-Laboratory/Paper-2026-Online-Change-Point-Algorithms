from functools import partial

import numpy as np
import polars as pl

from pathlib import Path

from evals.ground_truth import get_ground_truth_from_file, generate_ground_truth
from evals.scores import get_select_scores
from parsing.data_parsing import load_data_from_config
from utils.path_validation import confirm_dir_or_consult
from utils.read_data import load_signals
from utils.toml_utils import load_toml


def parse_eval_online(config_file):
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
        save_config = config_table['saves']
        eval_config = config_table['eval']
        # Load data
        eval_data_config = eval_config['data']
        match eval_data_config['what']:
            case 'csv':
                where = eval_data_config['where']
                if where['is-subdir']:
                    file_path = Path(
                        save_config['save-root'], where['dir'], where['filename'])
                else:
                    file_path = Path(where['dir'], where['filename'])
                df = pl.scan_csv(file_path.with_suffix('.csv')).collect()
            case x:
                raise NotImplementedError(f'No loading implementation for type {x}')
        print(df.head())
        # Get ground truth
        ## Optionally: save ground truth
        eval_ground = eval_config['ground']
        match eval_ground['what']:
            case 'save':
                dir = eval_ground['save']['dir']
                name = eval_ground['save']['name']
                eval_path = Path(dir, name)
                ground = get_ground_truth_from_file(eval_path)
            case 'generate':
                gen_config = eval_ground['generate']
                alg = gen_config['alg']
                gen_name = gen_config['name']
                # Load data
                data_config = config_table['data']
                time, data = load_data_from_config(data_config)
                ground = generate_ground_truth(data, alg=alg)
                # now check if you want to save this for later.
                gen_save = gen_config['saving']
                if gen_save['save']:
                    match gen_save['what']:
                        case 'npy' | 'numpy':
                            dir = gen_save['where']['dir']
                            name = gen_save['where']['save-name']
                            np.save(Path(dir, name), ground)
                        case x:
                            raise NotImplementedError(f'No implementation for saving generated ground truth for {x}')
            case x:
                raise ValueError(f'No option "{x}" for ground truth retrieval.')
        print(ground)
        # Evaluate
        metrics_config = eval_config['metrics']
        metric_names = metrics_config['scores']
        # get_select_scores(time, ground, )
        predictions = df.select('name', 'prediction').group_by('name')
        metric_fn = partial(get_select_scores, time, ground, metrics=metric_names)
        scores = [(name[0],)+tuple(metric_fn(vals.select('prediction').to_numpy(order='c').flatten()).values()) for name, vals in predictions]
        score_df = pl.DataFrame(scores, schema=['name']+metric_names, orient='row')
        print(score_df)
        # Save
        eval_save_config = eval_config['saving']
        if eval_save_config['save']:
            folder = eval_save_config['where']
            root = save_config['save-root'] if eval_save_config['is-subdir'] else ''
            save_name = eval_save_config['save-name']
            save_path = Path(root, folder, save_name)
            match eval_save_config['how']:
                case str(x) if x == 'csv':
                    score_df.write_csv(save_path.with_suffix('.csv'))
                case str(x) if x == 'tex-table':
                    raise NotImplementedError
                case None:
                    raise TypeError(f'[eval.saving] table is missing "how" key.')
    else:
        pass

def something_else(config_file):
    ## Process online models
    df = process_online_models(time, data, algs, ground)
    # results = run_online_models(time, data, algs)
    # plot_detection_online_models(time, data, results)
    # df = evaluate_online_models(time, results, ground)
    ## Process offline anomaly models
    anom_save_dir = config_table['save-root']
    confirm_dir_or_consult(Path(anom_save_dir))
    # anom_save_dir = Path(os.curdir, 'figures', '2025-04-07', 'signal-1-filtered')
    anom_save_name = Path('isolation_forest_fig')
    data_size = len(data)
    half_point = data_size//2
    safe = data[:half_point]
    unsafe = data[half_point:]
    anom_results = run_offline_anomaly_models(time, safe, unsafe, data, (AnomalyType.ISO_FOREST,))
    plot_detection_anomaly_models(time, data, anom_results, anom_save_dir, anom_save_name)
    ## Write data frame for LaTeX
    save_root = config_table['save-root']
    metric_table_config = config_table['metric-table']
    metric_root = metric_table_config['save-root']
    save_name = metric_table_config['save-name']
    save_folder = Path(save_root, metric_root)
    confirm_dir_or_consult(save_folder)
    df_reduced = df.loc[:, ['accuracy', 'precision', 'recall', 'f1 score', 'delay']]
    tex_table = format_frame_for_latex(df)
    tex_table_reduced = tex_table.loc[:, ['accuracy', 'precision', 'recall', 'f1 score', 'detection delay (ms)']]
    # save_folder = Path(os.curdir, 'figures', '2025-02-11', 'tables')
    # save_name = config_table['metric-table']['save-name']
    write_frame_to_latex(tex_table_reduced, save_name, save_folder)
    # Make radar chart
    # metric_radar_fig = plot_metric_scores_for_paper(df)
    # metric_fig_2 = plot_metrics_as_bar(df_reduced)
    metric_fig_3 = plot_metrics_as_radar_bar_for_paper(df_reduced)
    plt.savefig(Path(save_root, 'metric-radar-chart.png'), dpi=350)
    # metric_radar_fig.write_image(Path(save_root, 'metric_radar_chart.png'))
    # metric_radar_fig.write_image(Path(save_root, 'metric_radar_chart.pdf'))