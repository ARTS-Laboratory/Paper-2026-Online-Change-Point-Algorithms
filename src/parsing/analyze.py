from analysis.signal_plots import plot_signals, plot_offline_detections
from parsing.data_parsing import load_data_from_config
from utils.toml_utils import load_toml


def parse_analyze(config_file):
    """ """
    config_table = load_toml(config_file)
    # Get data
    data_config = config_table['data']
    time, data = load_data_from_config(data_config)
    save_root = None
    plot_signals(time, data, save_root)
    plot_offline_detections(time, data, save_root)
