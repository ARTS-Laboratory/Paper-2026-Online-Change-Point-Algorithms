import numpy as np

from change_point_algorithms.online_detection.cusum import cusum_alg_v0_rust_hybrid, cusum_alg_v0_generator_v1, \
    cusum_alg_generator


def generate_normal_points(mean: float, stddev: float, num_points: int):
    """
    :param mean:
    :param stddev:
    :param num_points:
    :return:
    """
    rng = np.random.default_rng()
    return rng.normal(mean, stddev, num_points)

def test_cusum_v0_with_deque_equivalent():
    safe_mean = 0.0
    safe_std_dev = 1.0
    threshold = 5.0
    alpha = 0.95
    num_unknowns = 1_000
    my_unknowns = generate_normal_points(safe_mean, safe_std_dev, num_unknowns)
    gen_1 = cusum_alg_generator(my_unknowns, safe_mean, safe_std_dev, threshold, alpha)
    gen_2 = cusum_alg_v0_generator_v1(my_unknowns, safe_mean, safe_std_dev, threshold, alpha)
    out = [val_1 == val_2 for val_1, val_2 in zip(gen_1, gen_2)]
    assert all(out)

class TestRustCusum:
    safe_mean = 0.0
    safe_std_dev = 1.0
    threshold = 5.0
    alpha = 0.95
    unsafe_mean = 50.0
    unsafe_std_dev = 1.0 # 2.0
    num_unknowns = 1_000
    num_to_pass = round(0.95 * num_unknowns)

    def test_cusum_v0_rust_hybrid_all_normal(self):
        my_unknowns = generate_normal_points(self.safe_mean, self.safe_std_dev * 0.01, self.num_unknowns)
        model_gen = cusum_alg_v0_rust_hybrid(
            my_unknowns, self.safe_mean, self.safe_std_dev, self.threshold, self.alpha)
        predictions = [item for item in model_gen]
        assert predictions.count(False) >= self.num_to_pass, f'Model predicted that {predictions.count(True)} were change points.'

    def test_cusum_v0_rust_hybrid_all_abnormal(self):
        my_unknowns = generate_normal_points(
            self.unsafe_mean, self.unsafe_std_dev, self.num_unknowns)
        model_gen = cusum_alg_v0_rust_hybrid(
            my_unknowns, self.safe_mean, self.safe_std_dev, self.threshold, self.alpha)
        predictions = [item for item in model_gen]
        # Skip first 10 to give cusum time to detect.
        assert predictions.count(True) >= self.num_to_pass, f'Model predicted that {predictions.count(False)} were change points.'
