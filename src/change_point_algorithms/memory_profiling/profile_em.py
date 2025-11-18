import math
import tracemalloc

import numpy as np

from change_point_algorithms.memory_profiling import profiling
from change_point_algorithms.online_detection.expect_Max import expectation_maximization_generator


def get_parameters():
    mean_1, var_1 = 0.0, 1.0
    mean_2, var_2 = 10.0, 2.0
    return mean_1, var_1, mean_2, var_2


def get_benchmark_vars():
    mean_1, var_1, mean_2, var_2 = get_parameters()
    safe_mean, unsafe_mean = 0.0, 50.0
    safe_var, unsafe_var = 1.0, 4.0
    safe_size, unsafe_size = 70, 30
    pi = 0.3
    epochs = 100
    num_unknowns = 400_000
    rng = np.random.default_rng()
    safe = rng.normal(mean_1, math.sqrt(var_1), safe_size)
    unsafe = rng.normal(mean_2, math.sqrt(var_2), unsafe_size)
    my_unknowns = rng.normal(mean_2, math.sqrt(var_2), num_unknowns)
    param_dict = {
        'safe_mean': safe_mean, 'unsafe_mean': unsafe_mean,
        'safe_var': safe_var, 'unsafe_var': unsafe_var, 'pi': pi}
    return param_dict, safe, unsafe, my_unknowns, epochs


def profile_em_from_python():
    """ Test memory utilization of expectation maximization with tracemalloc."""
    params, safe, unsafe, my_unknowns, epochs = get_benchmark_vars()
    mem_unit = 'MiB'
    tracemalloc.start(20)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage: {curr} B, peak of {peak} B')
    tracemalloc.reset_peak()
    print(tracemalloc.get_traced_memory())
    model_gen = expectation_maximization_generator(
        safe, unsafe, my_unknowns, params['safe_mean'], params['unsafe_mean'],
        params['safe_var'], params['unsafe_var'], params['pi'], epochs)
    profiling.profile_model_run(model_gen, mem_unit)
    tracemalloc.stop()