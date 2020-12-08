from abc import *

class ABSGradient(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, lr):
        self.lr = lr

    @abstractmethod
    def put(self, grain_weight, grain_bias):
        pass

    @abstractmethod
    def deltaWeight(self):
        pass

    @abstractmethod
    def deltaBias(self):
        pass

    @abstractmethod
    def reset(self):
        pass
