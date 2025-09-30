from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import Any


class ModelType(Enum):
    BOCPD = 'bocpd'
    EM = 'expectation maximization'
    CUSUM = 'cusum'
    GREY_MODEL = 'grey'
    NON_PARAMETRIC = 'nonparametric'


@dataclass
class DetectionAlgorithm:
    type: ModelType
    name: str
    with_progress: bool = False
    save_path: PathLike = None
    hyperparameters: Any = None

@dataclass
class DetectionAlgorithmV2:
    type: ModelType
    name: str
    hyperparameters: Any = None