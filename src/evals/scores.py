from collections.abc import Sequence

import numpy as np
import sklearn

def earliest_correct_score(time: np.ndarray, ground: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    """ """
    # Price is right score
    true_positive_indices: np.ndarray = np.logical_and(ground.astype(bool), predictions.astype(bool))
    if true_positive_indices.any():
        earliest_correct: float = time[true_positive_indices][0]
        first_ground: float = time[np.nonzero(ground.astype(bool))][0]
        delay: float = earliest_correct - first_ground
    else:
        earliest_correct: float = np.inf
        delay: float = np.inf
    return earliest_correct, delay

def get_default_scores(time, ground, predictions) -> dict[str, float]:
    """ Return metric score for predicted shock prediction given comparison."""
    # Calculate scores
    f1_score = sklearn.metrics.f1_score(ground, predictions)
    precision = sklearn.metrics.precision_score(ground, predictions)
    recall = sklearn.metrics.recall_score(ground, predictions)
    accuracy = sklearn.metrics.accuracy_score(ground, predictions)
    earliest_correct, delay = earliest_correct_score(time, ground, predictions)
    scores = {
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'detection delay': delay,
        'earliest correct': earliest_correct
    }
    return scores

def get_select_scores(time, ground, predictions, metrics=None) -> dict[str, float]:
    """ """
    return get_select_scores_v1(time, ground, predictions, metrics=metrics)

def get_select_scores_v1(time, ground, predictions, metrics: Sequence[str]) -> dict[str, float]:
    """ Return select metric scores for predicted shock prediction given comparison."""
    def calculate_metric(metric_name):
        match metric_name:
            case 'f1-score':
                return sklearn.metrics.f1_score(ground, predictions)
            case 'precision':
                return sklearn.metrics.precision_score(ground, predictions)
            case 'recall':
                return sklearn.metrics.recall_score(ground, predictions)
            case 'accuracy':
                return sklearn.metrics.accuracy_score(ground, predictions)
            case 'delay':
                return earliest_correct_score(time, ground, predictions)[1]
            case _:
                raise ValueError(f'Unknown metric {metric_name}')
    # as dictionary
    return {metric: calculate_metric(metric) for metric in metrics}

def get_select_scores_v2(time, ground, predictions, metrics: Sequence[str]) -> dict[str, float]:
    """ """
    metric_dict = dict()
    for metric in metrics:
        match metric:
            case 'f1-score':
                score = sklearn.metrics.f1_score(ground, predictions)
            case 'precision':
                score = sklearn.metrics.precision_score(ground, predictions)
            case 'recall':
                score = sklearn.metrics.recall_score(ground, predictions)
            case 'accuracy':
                score = sklearn.metrics.accuracy_score(ground, predictions)
            case 'delay':
                score = earliest_correct_score(time, ground, predictions)[1]
            case _:
                raise ValueError(f'Unknown metric {metric}')
        # print(f'Model scored {score} for metric "{metric}"')
        metric_dict[metric] = score
    return metric_dict

def print_scores(time, ground, predictions):
    """ Print metric scores for predicted shock prediction given comparison."""
    # Calculate scores
    f1_score = sklearn.metrics.f1_score(ground, predictions)
    precision = sklearn.metrics.precision_score(ground, predictions)
    recall = sklearn.metrics.recall_score(ground, predictions)
    accuracy = sklearn.metrics.accuracy_score(ground, predictions)
    # Print scores
    print(f'F1 score: {f1_score:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Accuracy: {accuracy:.3f}')
    # Print confusion matrix
    confusion = sklearn.metrics.confusion_matrix(ground, predictions, normalize='all')
    print(confusion)
    print(sklearn.metrics.classification_report(ground, predictions, digits=3))
    # Price is right score
    earliest_correct, delay = earliest_correct_score(time, ground, predictions)
    if earliest_correct != np.inf:
        print(f'Shock first correctly detected at time: {earliest_correct}')
        print(f'Detection delay: {delay}')
    else:
        print('No predictions aligned with ground truth.')
