import tracemalloc

import numpy as np

from change_point_algorithms.online_detection.grey_systems_model import grey_model_generator
from memory_profiling import profiling

def get_benchmark_vars():
    mean, std_dev = 0.0, 1.0
    window_size = 10
    c = 1.5
    c_ratio = 1.5
    data_size = 400_000# 10_000
    rng = np.random.default_rng()
    data = rng.normal(mean, std_dev, size=data_size)
    return data, window_size, c, c_ratio

def profile_grey_model_from_python():
    data, window_size, c, c_ratio = get_benchmark_vars()
    mem_unit = 'MiB'
    tracemalloc.start(20)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage: {curr} B, peak of {peak} B')
    tracemalloc.reset_peak()
    print(tracemalloc.get_traced_memory())
    model_gen = grey_model_generator(data, window_size, c=c, c_ratio=c_ratio)
    profiling.profile_model_run(model_gen, mem_unit)