import itertools
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import more_itertools
import numpy as np
import polars as pl

import Hyperparameters
from AnomalyAlgorithm import AnomalyType, AnomalyAlgorithm
from DetectionAlgorithm import DetectionAlgorithm, ModelType, DetectionAlgorithmV2
from model_runners.offline_anomaly_models import run_offline_anomaly_models_v2
from model_runners.online_models import run_online_models, ResultType, run_online_models_v2
from parsing.data_parsing import load_data_from_config
from utils.detection_arr_helpers import intervals_to_dense_arr
from utils.path_validation import confirm_dir_or_consult
from utils.read_data import load_signals
from utils.toml_utils import load_toml


def read_model_config(config_file: os.PathLike):
    """ Parse config file for models."""
    config_table = load_toml(config_file)
    default_save_path = Path(config_table['save-root'])
    confirm_dir_or_consult(default_save_path)
    models = config_table['models']
    algs = list()
    for model in models:
        hp = model['hyperparameters']
        model_type: str = model['type']
        model_name: str = model['name']
        if 'save-path' in model:
            save_dir = Path(model['save-path'])
            confirm_dir_or_consult(save_dir)
        else:
            save_dir = default_save_path
        m_save_path = Path(save_dir, model['save-name'])
        if 'show-progress' in model:
            with_progress = model['show-progress']
        else:
            with_progress = False
        match model_type:
            case 'bocpd':
                hyperparams = Hyperparameters.BOCPDHyperparams(
                    alpha=hp['alpha'].unwrap(), beta=hp['beta'].unwrap(),
                    mu=hp['mu'].unwrap(), kappa=hp['kappa'].unwrap(),
                    lamb=hp['lambda'].unwrap())
            case 'expectation maximization':
                hyperparams = Hyperparameters.EMHyperparams(
                        normal_data_size=hp['normal-data-size'].unwrap(),
                        abnormal_data_size=hp['abnormal-data-size'].unwrap(),
                        normal_mean=hp['normal-mean'].unwrap(),
                        abnormal_mean=hp['abnormal-mean'].unwrap(),
                        normal_var=hp['normal-variance'].unwrap(),
                        abnormal_var=hp['abnormal-variance'].unwrap(),
                        pi=hp['pi'].unwrap(), epochs=hp['epochs'].unwrap())
            case 'cusum':
                hyperparams = Hyperparameters.CUSUMHyperparams(
                    mean=hp['mean'].unwrap(),
                    std_dev=hp['standard-deviation'].unwrap(),
                    h=hp['h'].unwrap(),
                    alpha=hp['alpha'].unwrap())
            case 'grey':
                hyperparams = Hyperparameters.GreyHyperparams(
                    window_size=hp['window-size'].unwrap(),
                    critical_value=hp['critical-value'].unwrap(),
                    critical_ratio_value=hp['critical-ratio-value'].unwrap(),
                    alpha=hp['alpha'].unwrap())
            case 'nonparametric':
                hyperparams = Hyperparameters.NonparametricHyperparams(
                    window_size=hp['window-size'].unwrap(),
                    critical_value=hp['critical-value'].unwrap(),
                    alpha=hp['alpha'].unwrap()
                    )
            case _:
                raise NotImplementedError
        alg = DetectionAlgorithm(
            type=ModelType(model_type),
            name=model_name, with_progress=with_progress,
            save_path=m_save_path, hyperparameters=hyperparams)
        algs.append(alg)
    return algs

def get_model_hyperparameters(model_type: ModelType, hp: Mapping) -> object:
    """ Get hyperparameters from config.

        :param model_type: Model type.
        :param hp: Hyperparameter section of config for model.
    """
    match model_type:
        case ModelType.BOCPD:
            return Hyperparameters.BOCPDHyperparams(
                alpha=hp['alpha'].unwrap(), beta=hp['beta'].unwrap(),
                mu=hp['mu'].unwrap(), kappa=hp['kappa'].unwrap(),
                lamb=hp['lambda'].unwrap())
        case ModelType.EM:
            return Hyperparameters.EMHyperparams(
                normal_data_size=hp['normal-data-size'].unwrap(),
                abnormal_data_size=hp['abnormal-data-size'].unwrap(),
                normal_mean=hp['normal-mean'].unwrap(),
                abnormal_mean=hp['abnormal-mean'].unwrap(),
                normal_var=hp['normal-variance'].unwrap(),
                abnormal_var=hp['abnormal-variance'].unwrap(),
                pi=hp['pi'].unwrap(), epochs=hp['epochs'].unwrap())
        case ModelType.CUSUM:
            return Hyperparameters.CUSUMHyperparams(
                mean=hp['mean'].unwrap(),
                std_dev=hp['standard-deviation'].unwrap(),
                h=hp['h'].unwrap(),
                alpha=hp['alpha'].unwrap())
        case ModelType.GREY_MODEL:
            return Hyperparameters.GreyHyperparams(
                window_size=hp['window-size'].unwrap(),
                critical_value=hp['critical-value'].unwrap(),
                critical_ratio_value=hp['critical-ratio-value'].unwrap(),
                alpha=hp['alpha'].unwrap())
        case ModelType.NON_PARAMETRIC:
            return Hyperparameters.NonparametricHyperparams(
                window_size=hp['window-size'].unwrap(),
                critical_value=hp['critical-value'].unwrap(),
                alpha=hp['alpha'].unwrap()
                )

def get_models_from_config(models_config: Iterable[dict]) -> dict[str, DetectionAlgorithmV2]:
    """ Create dictionary of detection algorithms from config file."""
    algs = dict()
    for model in models_config:
        try:
            model_type = ModelType(model['type'])
        except KeyError:
            raise ValueError(f"Invalid model type: {model['type']}")
        name = model['name']
        hyperparameters = get_model_hyperparameters(model_type, model['hyperparameters'])
        alg = DetectionAlgorithmV2(model_type, name, hyperparameters)
        algs[name] = alg
    return algs

def get_anomaly_model_hyperparameters(model_type: AnomalyType, hp: Mapping) -> dataclass:
    """ Get hyperparameters from config.

        :param model_type: Model type.
        :param hp: Hyperparameter section of config for model.
    """
    match model_type:
        case AnomalyType.SVM | AnomalyType.ISO_FOREST:
            safe_vals = hp['safe']
            unsafe_vals = hp['unsafe']
            safe = [Hyperparameters.NormalDistParams(
                Hyperparameters.Normal(
                    item['mean'].unwrap(), item['std_dev'].unwrap()),
                item['num'].unwrap()) for item in safe_vals]
            unsafe = [Hyperparameters.NormalDistParams(
                Hyperparameters.Normal(
                    item['mean'].unwrap(), item['std_dev'].unwrap()),
                item['num'].unwrap()) for item
                    in unsafe_vals]
            return Hyperparameters.SafeUnsafeNormalMixtureHyperparams(safe, unsafe)

def get_anomaly_models_from_config(model_config: Iterable[dict]) -> dict:
    """ """
    algs = dict()
    for model in model_config:
        try:
            model_type = AnomalyType(model['type'])
        except KeyError:
            raise ValueError(f"Invalid anomaly model type: {model['type']}")
        name = model['name']
        hyperparameters = get_anomaly_model_hyperparameters(model_type, model['hyperparameters'])
        alg = AnomalyAlgorithm(model_type, name, hyperparameters)
        algs[name] = alg
    return algs

def parse_run_online(config_file):
    # set_rc_params()
    config_table = load_toml(config_file)
    version = config_table['version']
    if version == '1':
        parse_run_online_for_v1(config_table)
    elif version == '2':
        parse_run_online_for_v2(config_table)

def parse_run_online_for_v1(config_table):
    """ """
    file_path = Path(config_table['file-path'])
    time, data = load_signals(file_path)
    algs = read_model_config(config_file)
    results = run_online_models(time, data, algs)
    # convert to dataframe of time, model_name, prediction
    df = online_model_results_to_polars(time, results)
    print(df)

def parse_run_online_for_v2(config_table):
    """ """
    # Get data
    data_config = config_table['data']
    time, data = load_data_from_config(data_config)
    run_config = config_table['run']
    # Make algorithm objects
    algs = get_models_from_config(config_table['models'])
    # Extra options
    show_progress = run_config['show-progress']
    # Choose algorithms to run
    if run_config['run_all']:
        chosen_algs = algs.values()
    else:
        names = run_config['model_names']
        chosen_algs = (algs[name] for name in names)
    results = run_online_models_v2(time, data, chosen_algs, with_progress=show_progress)
    # Choose anomaly algorithms to run if enabled
    anomaly_algs = get_anomaly_models_from_config(config_table['anomaly-models'])
    # Choose algorithms to run
    if run_config['run_all']:
        chosen_algs = anomaly_algs.values()
    else:
        names = run_config['anomaly-model-names']
        chosen_algs = (anomaly_algs[name] for name in names)
    anomaly_results = run_offline_anomaly_models_v2(time, data, chosen_algs, with_progress=show_progress)
    # convert to dataframe of time, model_name, prediction
    df = online_model_results_to_polars(time, results)
    print(df)
    anomaly_df = online_model_results_to_polars(time, anomaly_results)
    print(anomaly_df)
    res_df = pl.concat([df, anomaly_df])
    # Saving results
    saves_config = config_table['saves']  # overall saving information
    run_save_config = run_config['saving']  # run specific saving information
    parse_run_saving(saves_config, run_save_config, res_df)


# def parse_run_data_config(data_config):
#     """ """
#     if data_config['what'] == 'array':
#         if 'dir' in data_config['where']:
#             file_path = Path(data_config['where']['dir'], data_config['where']['filename'])
#         else:
#             file_path = Path(data_config['where']['filename'])
#         time, data = load_signals(file_path)
#     else:
#         raise NotImplementedError(f"No implementation for data of type {data_config['what']}")
#     return time, data

# def parse_run_running():
#     """ """

def parse_run_saving(saves_config, run_save_config, df: pl.DataFrame):
    """ Parse 'saving' table of run config.

        :param saves_config: Config table of overall saving information
        :param run_save_config: Config table of saving information for run specifically.
        :param df: DataFrame of data to save.
    """
    if run_save_config['save']:
        folder = run_save_config.get('where', '')
        if run_save_config['is-subdir']:
            save_dir = Path(saves_config['save-root'], folder)
        else:
            save_dir = Path(folder)
        if run_save_config['save-as'] == 'csv':
            confirm_dir_or_consult(save_dir)
            save_path = Path(save_dir, run_save_config['save-name'])
            df.write_csv(save_path.with_suffix('.csv'))
        else:
            raise NotImplementedError(f"No implementation for saving {run_save_config['save-as']}")

def online_model_results_to_polars(time: np.ndarray, results: Iterable[ResultType]):
    """ """
    return online_model_results_to_polars_v1(time, results)

def online_model_results_to_polars_v0(time: np.ndarray, results: Iterable[ResultType]):
    """ """
    # arr_size = len(time)
    tuple_list = []
    for result in results:
        alg, shock_region, non_shock_region = result
        name = alg.name
        predictions = intervals_to_dense_arr(time, shock_region, non_shock_region)
        change_point_times = set((start for start, end in itertools.chain(shock_region, non_shock_region)))
        # change_points = [point in change_point_times for point in time]
        tuples = ((name, time_point, prediction, time_point in change_point_times) for time_point, prediction in zip(time, predictions))
        # tuples = [(name, time_point, prediction) for time_point, prediction in zip(time, predictions)]
        tuple_list.extend(tuples)
    df = pl.DataFrame(tuple_list, ['name', 'time', 'prediction', 'is_change_point'], orient='row')
    return df

def online_model_results_to_polars_v1(time: np.ndarray, results: Iterable[ResultType]):
    """ """
    # arr_size = len(time)
    def results_to_polars_helper(result):
        alg, shock_region, non_shock_region = result
        name = alg.name
        predictions = intervals_to_dense_arr(time, shock_region, non_shock_region)
        change_point_times = set((start for start, end in itertools.chain(shock_region, non_shock_region)))
        # change_point_times = more_itertools.unique_everseen(start for start, end in itertools.chain(shock_region, non_shock_region))
        # change_points = [point in change_point_times for point in time]
        return ((name, time_point, prediction, time_point in change_point_times) for time_point, prediction in zip(time, predictions))
    tuple_list = list(
        more_itertools.flatten(
            results_to_polars_helper(result) for result in results))
    df = pl.DataFrame(tuple_list, ['name', 'time', 'prediction', 'is_change_point'], orient='row')
    return df
