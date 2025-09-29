from pathlib import Path

from utils.read_data import load_signals


def load_data_from_config(data_config: dict):
    """ """
    if data_config['what'] == 'array':
        if 'dir' in data_config['where']:
            file_path = Path(data_config['where']['dir'], data_config['where']['filename'])
        else:
            file_path = Path(data_config['where']['filename'])
        return load_signals(file_path)
        # time, data = load_signals(file_path)
        # return time, data
    else:
        raise NotImplementedError(f"No implementation for data of type {data_config['what']}")

# def load_csv_
