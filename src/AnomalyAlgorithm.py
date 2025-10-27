from dataclasses import dataclass
from enum import Enum


class AnomalyType(Enum):
    SVM = 'svm'
    ISO_FOREST = 'isolation-forest'

@dataclass
class AnomalyAlgorithm:
    type: AnomalyType
    name: str
    hyperparameters: object | None = None
