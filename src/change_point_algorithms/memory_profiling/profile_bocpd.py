import tracemalloc

from change_point_algorithms.online_detection.bocpd import bocpd_generator, bocpd_rust_hybrid
from change_point_algorithms.memory_profiling import profiling
from change_point_algorithms.memory_profiling.generate_data import generate_normal_data


def get_benchmark_vars():
    data_size = 400_000 # 10_000
    # rng = np.random.default_rng()
    alpha, beta, mu, kappa, lamb = 1.0, 1.0, 0.0, 2.0, 2.0
    std_dev = 1.0
    # points = rng.normal(mu, std_dev, size=vec_size)
    points = generate_normal_data(mu, std_dev, data_size)
    return points, alpha, beta, mu, kappa, lamb

def profile_bocpd_from_python():
    """ Get memory utilization of bocpd with tracemalloc."""
    points, alpha, beta, mu, kappa, lamb = get_benchmark_vars()
    mem_unit = 'MiB'
    print('\n')
    tracemalloc.start(20)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage: {curr} B, peak of {peak} B')
    tracemalloc.reset_peak()
    print(tracemalloc.get_traced_memory())
    bocpd_model_gen = bocpd_generator(
        points, mu, kappa, alpha, beta, lamb)
    profiling.profile_model_run(bocpd_model_gen, mem_unit)
    tracemalloc.stop()

def profile_bocpd_from_rust():
    """ Get memory utilization of bocpd with tracemalloc."""
    points, alpha, beta, mu, kappa, lamb = get_benchmark_vars()
    mem_unit = 'MiB'
    tracemalloc.start(20)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage: {curr} B, peak of {peak} B')
    tracemalloc.reset_peak()
    bocpd_model_gen = bocpd_rust_hybrid(
        points, mu, kappa, alpha, beta, lamb)
    curr, peak = tracemalloc.get_traced_memory()
    profiling.profile_model_run(bocpd_model_gen, mem_unit)
    tracemalloc.stop()
