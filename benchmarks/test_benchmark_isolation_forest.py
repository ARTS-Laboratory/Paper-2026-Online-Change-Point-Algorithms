import numpy as np

from benchmarks.benchmark_helpers import benchmark_generator
from change_point_algorithms.model_runners.offline_anomaly_models import run_offline_anomaly_models
from change_point_algorithms.AnomalyAlgorithm import AnomalyType


def get_benchmark_vars():
    data_size = 400_000
    half_data_size = data_size // 2
    time_vec = np.linspace(0, 10, data_size)
    safe_mean, unsafe_mean = 0.0, 50.0
    safe_dev, unsafe_dev = 1.0, 2.0
    safe_size, unsafe_size = half_data_size, half_data_size
    rng = np.random.default_rng()
    safe = rng.normal(safe_mean, safe_dev, safe_size)
    unsafe = rng.normal(unsafe_mean, unsafe_dev, unsafe_size)
    data = np.concatenate([
        rng.normal(safe_mean, safe_dev, safe_size),
        rng.normal(unsafe_mean, unsafe_dev, unsafe_size)
    ])
    return time_vec, safe, unsafe, data


def test_benchmark_isolation_forest_from_python(benchmark):
    time_vec, safe, unsafe, data = get_benchmark_vars()
    anom_results = run_offline_anomaly_models(time_vec, safe, unsafe, data, (AnomalyType.ISO_FOREST,))
    out = benchmark(benchmark_generator, run_offline_anomaly_models, time_vec, safe, unsafe, data, (AnomalyType.ISO_FOREST,))
    assert out is not None
