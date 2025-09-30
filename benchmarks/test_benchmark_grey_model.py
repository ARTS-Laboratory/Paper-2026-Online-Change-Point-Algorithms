from benchmarks.benchmark_helpers import benchmark_generator
from benchmarks.generate_data import generate_normal_data
from change_point_algorithms.online_detection.grey_systems_model import grey_model_generator


def get_benchmark_vars():
    mean, std_dev = 0.0, 1.0
    window_size = 10
    c = 1.5
    c_ratio = 1.5
    data_size = 400_000# 10_000
    data = generate_normal_data(mean, std_dev, data_size)
    return data, window_size, c, c_ratio

def test_benchmark_grey_model_from_python(benchmark):
    data, window_size, c, c_ratio = get_benchmark_vars()
    # model_gen = grey_model_generator(
    #     data, window_size, c=c, c_ratio=c_ratio)
    benchmark(benchmark_generator, grey_model_generator, data, window_size, c=c, c_ratio=c_ratio)
    # benchmark(benchmark_generator, model_gen)
    # benchmark(lambda: [item for item in model_gen])
