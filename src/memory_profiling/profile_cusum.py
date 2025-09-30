import tracemalloc

from change_point_algorithms.online_detection.cusum import cusum_alg_generator, cusum_alg_v0_generator_v1
from memory_profiling.generate_data import generate_normal_data
from memory_profiling.profiling import profile_model_run


def get_benchmark_vars():
    mean = 20.0
    std_dev = 5.0
    h = 5
    alpha = 0.95
    data_size = 400_000 # 10_000
    # rng = np.random.default_rng()
    # data = rng.normal(mean, std_dev, size=data_size)
    data = generate_normal_data(mean, std_dev, data_size)
    return data, mean, std_dev, h, alpha

def profile_cusum_alg():
    """ """
    data, mean, std_dev, h, alpha = get_benchmark_vars()
    mem_unit = 'MiB'
    print('\n')
    tracemalloc.start(20)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage: {curr} B, peak of {peak} B')
    tracemalloc.reset_peak()
    print(tracemalloc.get_traced_memory())
    model_gen = cusum_alg_generator(data, mean, std_dev, h, alpha)
    profile_model_run(model_gen, mem_unit)
    tracemalloc.stop()

def profile_cusum_alg_v1():
    """ """
    data, mean, std_dev, h, alpha = get_benchmark_vars()
    mem_unit = 'MiB'
    tracemalloc.start(20)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage: {curr} B, peak of {peak} B')
    tracemalloc.reset_peak()
    print(tracemalloc.get_traced_memory())
    model_gen = cusum_alg_v0_generator_v1(data, mean, std_dev, h, alpha)
    profile_model_run(model_gen, mem_unit)
    tracemalloc.stop()
