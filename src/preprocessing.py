import numpy as np
import scipy.stats as st

class Preprocessing:
    @staticmethod
    def build_design_matrix(X: np.ndarray) -> np.ndarray:
        """Adiciona uma coluna de uns à matriz X para representar o recurso de bias."""
        return np.hstack((np.ones((X.shape[0], 1)), X))
