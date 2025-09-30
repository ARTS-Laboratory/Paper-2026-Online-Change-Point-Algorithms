import math

import numpy as np

from benchmarks.benchmark_helpers import benchmark_generator
from benchmarks.generate_data import generate_normal_data
from change_point_algorithms.online_detection.expect_Max import expectation_maximization_generator, em_rust_hybrid


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

def get_benchmark_vars_2():
    mean_1, var_1, mean_2, var_2 = get_parameters()
    safe_mean, unsafe_mean = 0.0, 50.0
    safe_std_dev, unsafe_std_dev = 1.0, 2.0
    safe_size, unsafe_size = 70, 30
    pi = 0.3
    epochs = 100
    num_unknowns = 400_000
    my_unknowns = generate_normal_data(mean_1, math.sqrt(var_1), num_unknowns)
    param_dict = {
        'safe_mean': safe_mean, 'unsafe_mean': unsafe_mean,
        'safe_std_dev': safe_std_dev, 'unsafe_std_dev': unsafe_std_dev,
        'num_safe': safe_size, 'num_unsafe': unsafe_size,'pi': pi}
    return param_dict, my_unknowns, epochs

def test_benchmark_em_model_from_python(benchmark):
    params, safe, unsafe, my_unknowns, epochs = get_benchmark_vars()
    benchmark(
        benchmark_generator, expectation_maximization_generator,
        safe, unsafe, my_unknowns, params['safe_mean'], params['unsafe_mean'],
        params['safe_var'], params['unsafe_var'], params['pi'], epochs)

# def test_benchmark_em_from_rust_hybrid(benchmark):
#     params, my_unknowns, epochs = get_benchmark_vars_2()
#     benchmark(
#         benchmark_generator, em_rust_hybrid, my_unknowns,
#         params['safe_mean'], params['safe_std_dev'],
#         params['num_safe'], params['unsafe_mean'], params['unsafe_std_dev'],
#         params['num_unsafe'], params['pi'], epochs=epochs)

def test_benchmark_em_from_rust_hybrid_with_early_stop(benchmark):
    params, my_unknowns, epochs = get_benchmark_vars_2()
    benchmark(
        benchmark_generator, em_rust_hybrid, my_unknowns,
        params['safe_mean'], params['safe_std_dev'],
        params['num_safe'], params['unsafe_mean'], params['unsafe_std_dev'],
        params['num_unsafe'], params['pi'], epochs=epochs, early_stopping=True)
