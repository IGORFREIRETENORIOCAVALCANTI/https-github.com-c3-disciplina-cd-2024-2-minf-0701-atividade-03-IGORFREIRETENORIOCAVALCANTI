from tokenize import Double
import numpy as np
import scipy.stats as st
import abc
import src.models as models

class OptimizerStrategy(abc.ABC):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    @abc.abstractmethod
    def update_model(self, X, y, model):
        """Implement Update Weigth Strategy"""

class SteepestDescentMethod(OptimizerStrategy):
    def __init__(self, learning_rate: float):
        super().__init__(learning_rate)

    def update_model(self, X: np.ndarray, y: np.ndarray, model: models.Model):
        gradient = (2.0/len(X)) * (np.linalg.multi_dot([X.T, X, model.w]) - np.dot(X.T, y))
        model.w -= self.learning_rate * gradient


class NewtonsMethod(OptimizerStrategy):
    def __init__(self, learning_rate: float):
        super().__init__(learning_rate)

    def update_model(self, X: np.ndarray, y: np.ndarray, model: models.Model):
        XTX = X.T @ X + np.eye(X.shape[1]) * 1e-5
        model.w = (1 - self.learning_rate) * model.w + self.learning_rate * np.linalg.multi_dot([np.linalg.inv(XTX), X.T, y])