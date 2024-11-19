import numpy as np
import scipy.stats as st
from typing import Tuple

class DataGenerator:
    def __init__(self, n: int, w: np.ndarray, x_min: float, x_max: float, std: float = 1.0):
        self.n = n
        self.w = w
        self.x_min = x_min
        self.x_max = x_max
        self.std = std

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X = np.linspace(self.x_min, self.x_max, self.n).reshape(-1, 1)
        X = np.hstack((np.ones((self.n, 1)), X))
        y = X @ self.w + np.random.normal(0, self.std, size=(self.n, 1))
        return X, y