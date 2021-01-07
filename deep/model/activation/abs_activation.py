from abc import *

class ABSActivation(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, output):
        pass

    @abstractmethod
    def backward(self, error, y):
        pass
