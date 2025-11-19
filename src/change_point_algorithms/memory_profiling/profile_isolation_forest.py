import tracemalloc

import numpy as np

from change_point_algorithms.memory_profiling.profiling import get_scalar_unit
from change_point_algorithms.model_runners.offline_anomaly_models import run_offline_anomaly_models
from change_point_algorithms.AnomalyAlgorithm import AnomalyType


def get_benchmark_vars():
    data_size = 10_000
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


def profile_isolation_forest_from_python():
    time_vec, safe, unsafe, data = get_benchmark_vars()
    mem_unit = 'MiB'
    mem_scalar = get_scalar_unit(mem_unit)
    tracemalloc.start(20)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage: {curr / mem_scalar} {mem_unit}, peak of {peak / mem_scalar} {mem_unit}')
    tracemalloc.reset_peak()
    print(tracemalloc.get_traced_memory())
    anom_results = run_offline_anomaly_models(time_vec, safe, unsafe, data, (AnomalyType.ISO_FOREST,))
    curr, peak = tracemalloc.get_traced_memory()
    print(f'End of function current memory usage: {curr / mem_scalar:.4} {mem_unit}, peak of {peak / mem_scalar:.4} {mem_unit}')
    tracemalloc.stop()
