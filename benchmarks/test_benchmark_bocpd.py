from benchmarks.benchmark_helpers import benchmark_generator
from benchmarks.generate_data import generate_normal_data
from change_point_algorithms.online_detection.bocpd import bocpd_generator, bocpd_rust_hybrid


def get_benchmark_vars():
    data_size = 10_000
    # rng = np.random.default_rng()
    alpha, beta, mu, kappa, lamb = 1.0, 1.0, 0.0, 2.0, 2.0
    std_dev = 1.0
    # points = rng.normal(mu, std_dev, size=vec_size)
    points = generate_normal_data(mu, std_dev, data_size)
    return points, alpha, beta, mu, kappa, lamb

def test_benchmark_bocpd_from_python(benchmark):
    points, alpha, beta, mu, kappa, lamb = get_benchmark_vars()
    # bocpd_model_gen = bocpd_generator(
    #     points, mu, kappa, alpha, beta, lamb)
    out = benchmark(benchmark_generator, bocpd_generator, points, mu, kappa, alpha, beta, lamb)
    assert out is not None

def test_benchmark_bocpd_rust_hybrid(benchmark):
    points, alpha, beta, mu, kappa, lamb = get_benchmark_vars()
    # bocpd_model_gen = bocpd_rust_hybrid(
    #     points, mu, kappa, alpha, beta, lamb)
    out = benchmark(
        benchmark_generator,
        bocpd_rust_hybrid, points, mu, kappa, alpha, beta, lamb)
    assert out is not None
