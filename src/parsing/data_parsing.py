from pathlib import Path

from utils.read_data import load_signals

DIR = 'dir'

FILENAME = 'filename'

WHAT = 'what'

DATA_WHERE = 'where'


def load_data_from_config(data_config: dict):
    """ """
    if data_config[WHAT] == 'array':
        if DIR in data_config[DATA_WHERE]:
            file_path = Path(data_config[DATA_WHERE][DIR], data_config[DATA_WHERE][FILENAME])
        else:
            file_path = Path(data_config[DATA_WHERE][FILENAME])
        return load_signals(file_path)
        # time, data = load_signals(file_path)
        # return time, data
    else:
        raise NotImplementedError(f"No implementation for data of type {data_config[WHAT]}")

# def load_csv_
