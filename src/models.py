import numpy as np
import scipy.stats as st
import abc
import src.optimizers as opt

class Model(abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        
    
    @abc.abstractmethod
    def predict() -> np.ndarray:
        """Implement the predict method"""
        
    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value

class LinearModel(Model):
    def __init__(self):
        super().__init__()
        self._w = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._w is None:
            raise ValueError("Weights must be set before prediction.")
        X = X[:, 1:]
        return X @ self.w