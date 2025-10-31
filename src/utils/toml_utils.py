""" Thompson 2022"""
import os

import tomlkit


def load_toml(filename: os.PathLike) -> tomlkit.TOMLDocument:
    """

    :param filename: Path to TOML file.
    :return: TOML Document located at filepath
    """
    with open(filename, 'r') as toml_file:
        data = tomlkit.load(toml_file)
    return data
