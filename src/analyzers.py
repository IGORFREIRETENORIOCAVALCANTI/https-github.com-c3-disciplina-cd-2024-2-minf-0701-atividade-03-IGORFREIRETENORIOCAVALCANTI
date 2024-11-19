import src.algorithms as al
import abc 
import matplotlib.pyplot as plt
import numpy as np

class AlgorithmAnalyzer(abc.ABC):

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def notify_started(self, alg: al.Algorithm):
        pass

    @abc.abstractmethod
    def notify_finished(self, alg: al.Algorithm):
        pass

    @abc.abstractmethod
    def notify_iteration(self, alg: al.Algorithm):
        pass

class PlotterAlgorithmObserver(AlgorithmAnalyzer):
    def __init__(self):
        super().__init__()
        self.iterations = []
        self.errors = []
        self.weights = []

    def notify_started(self, alg):
        self.iterations.clear()
        self.errors.clear()
        self.weights.clear()

    def notify_finished(self, alg):
        pass

    def notify_iteration(self, alg):
        self.iterations.append(alg.iteration)
        self.errors.append(np.mean(alg.errors))
        self.weights.append(alg.w)

    def plot(self, weights):
        fig, ax = plt.subplots()
        ax.plot(self.iterations, self.errors, label='Error')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error')
        ax.legend()
        plt.show()