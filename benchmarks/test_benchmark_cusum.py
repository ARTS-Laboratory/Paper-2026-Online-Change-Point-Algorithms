import numpy as np

from benchmarks.benchmark_helpers import benchmark_generator
from benchmarks.generate_data import generate_normal_data
from change_point_algorithms.online_detection.cusum import cusum_alg_generator, get_cusum_from_generator, \
    cusum_alg_v0_rust_hybrid, cusum_alg_v1_generator, cusum_alg_v1_rust_hybrid


def get_benchmark_vars():
    mean = 20.0
    std_dev = 5.0
    h = 5
    alpha = 0.95
    data_size = 10_000
    # rng = np.random.default_rng()
    # data = rng.normal(mean, std_dev, size=data_size)
    data = generate_normal_data(mean, std_dev, data_size)
    return data, mean, std_dev, h, alpha

def test_benchmark_cusum_alg(benchmark):
    data, mean, std_dev, h, alpha = get_benchmark_vars()
    # model_gen = cusum_alg_generator(data, mean, std_dev, h, alpha)
    benchmark(benchmark_generator, cusum_alg_generator, data, mean, std_dev, h, alpha)
    # benchmark(benchmark_generator, model_gen)
    # benchmark(lambda: [item for item in model_gen])


# def test_benchmark_get_cusum_generator_alg_v0(benchmark):
#     data, mean, std_dev, h, alpha = get_benchmark_vars()
#     time = np.arange(len(data))
#     version = 'v0'
#     benchmark(
#         get_cusum_from_generator, time, data, mean, std_dev, h, alpha,
#         with_progress=True, version=version)

def test_benchmark_get_cusum_generator_alg_v0_rust_hybrid(benchmark):
    data, mean, std_dev, h, alpha = get_benchmark_vars()
    # model_gen = cusum_alg_v0_rust_hybrid(data, mean, std_dev, h, alpha)
    benchmark(benchmark_generator, cusum_alg_v0_rust_hybrid, data, mean, std_dev, h, alpha)
    # benchmark(benchmark_generator, model_gen)
    # benchmark(lambda: [item for item in model_gen])

def test_benchmark_cusum_alg_v1(benchmark):
    data, mean, std_dev, h, alpha = get_benchmark_vars()
    # model_gen = cusum_alg_v1_generator(data, mean, std_dev, h, alpha)
    benchmark(benchmark_generator, cusum_alg_v1_generator, data, mean, std_dev, h, alpha)
    # benchmark(benchmark_generator, model_gen)
    # benchmark(lambda: [item for item in model_gen])

def test_benchmark_get_cusum_generator_alg_v1_rust_hybrid(benchmark):
    data, mean, std_dev, h, alpha = get_benchmark_vars()
    # model_gen = cusum_alg_v1_rust_hybrid(data, mean, std_dev, h, alpha)
    benchmark(benchmark_generator, cusum_alg_v1_rust_hybrid, data, mean, std_dev, h, alpha)
    # benchmark(benchmark_generator, model_gen)
    # benchmark(lambda: [item for item in model_gen])
