import numpy as np
from change_point_algorithms import _change_point_algorithms

from change_point_algorithms.online_detection.bocpd import bocpd_rust_hybrid, bocpd_generator


def get_parameters():
    alpha, beta, mu, kappa, lamb = 1.0, 1.0, 0.0, 2.0, 2.0
    # std_dev = 1.0
    # points = rng.normal(mu, std_dev, size=vec_size)
    return alpha, beta, mu, kappa, lamb

def generate_normal_points(mean: float, stddev: float, num_points: int):
    """
    :param mean:
    :param stddev:
    :param num_points:
    :return:
    """
    rng = np.random.default_rng()
    return rng.normal(mean, stddev, num_points)


class TestRustModels:
    alpha, beta, mu, kappa, lamb = get_parameters()
    num_unknowns = 200
    num_to_pass = round(0.95 * num_unknowns)
    safe_mean = mu
    safe_stddev = 1.0
    # safe_stddev: float = (beta * (kappa + 1.0)) / (kappa * alpha)
    unsafe_mean = 100.0
    unsafe_stddev = 1.0


    def test_bocpd_rust_hybrid_all_normal(self):
        my_unknowns = generate_normal_points(
            self.safe_mean, self.safe_stddev, self.num_unknowns)
        model_gen = bocpd_rust_hybrid(
            my_unknowns, self.mu, self.kappa, self.alpha, self.beta, self.lamb)
        predictions = [item for item in model_gen]
        assert predictions.count(False) >= self.num_to_pass, f'Model predicted that {predictions.count(True)} were change points.'

    def test_bocpd_rust_hybrid_all_abnormal(self):
        my_unknowns = generate_normal_points(
            self.unsafe_mean, self.unsafe_stddev, self.num_unknowns)
        model_gen = bocpd_rust_hybrid(
            my_unknowns, self.mu, self.kappa, self.alpha, self.beta, self.lamb)
        predictions = [item for item in model_gen]
        assert predictions.count(True) >= self.num_to_pass, f'Model predicted that {predictions.count(False)} were change points.'

    def test_bocpd_rust_hybrid_all_normal_probabilities(self):
        last_prob = 1.0
        num_unknowns = 100_000
        prob_threshold = 0.05
        my_unknowns = generate_normal_points(
            self.safe_mean, self.safe_stddev, num_unknowns)
        model = _change_point_algorithms.BocpdModel(self.alpha, self.beta, self.mu, self.kappa, True, threshold=1e-8)
        predictions = []
        for idx, event in enumerate(my_unknowns):
            model.update(event, 1000.0)
            probability: float = model.predict(event)
            probability *= last_prob
            assert probability <= 1.0, f'Probability at idx {idx}: {probability}'
            is_attack = probability <= prob_threshold
            last_prob = probability if not is_attack else 1.0
            predictions.append(is_attack)
        assert predictions.count(False) >= self.num_to_pass, f'Model predicted that {predictions.count(True)} were change points.'


class TestPythonModels:
    alpha, beta, mu, kappa, lamb = get_parameters()
    num_unknowns = 200
    num_to_pass = round(0.95 * num_unknowns)
    safe_mean = mu
    safe_stddev = 1.0
    # safe_stddev: float = (beta * (kappa + 1.0)) / (kappa * alpha)
    unsafe_mean = 100.0
    unsafe_stddev = 1.0

    def test_bocpd_python_all_normal(self):
        my_unknowns = generate_normal_points(
            self.safe_mean, self.safe_stddev, self.num_unknowns)
        model_gen = bocpd_generator(
            my_unknowns, self.mu, self.kappa, self.alpha, self.beta, self.lamb)
        predictions = [item for item in model_gen]
        assert predictions.count(False) >= self.num_to_pass, f'Model predicted that {predictions.count(True)} were change points.'

    def test_bocpd_python_all_abnormal(self):
        my_unknowns = generate_normal_points(
            self.unsafe_mean, self.unsafe_stddev, self.num_unknowns)
        model_gen = bocpd_generator(
            my_unknowns, self.mu, self.kappa, self.alpha, self.beta, self.lamb)
        predictions = [item for item in model_gen]
        assert predictions.count(True) >= self.num_to_pass, f'Model predicted that {predictions.count(False)} were change points.'
