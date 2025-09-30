import numpy as np

def generate_normal_data(mean: float, std_dev: float, num_samples: int):
    """ """
    rng = np.random.default_rng()
    return rng.normal(mean, std_dev, size=num_samples)