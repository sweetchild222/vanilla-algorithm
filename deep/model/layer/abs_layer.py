from abc import *

class ABSLayer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, backward_layer):

        self.input_shape = None
        self.backward_layer = backward_layer
        self.forward_layer = None

        if backward_layer is not None:
            self.input_shape = backward_layer.outputShape()
            backward_layer.forward_layer = self


    def forwardLayer(self):
        return self.forward_layer


    def backwardLayer(self):
        return self.backward_layer


    @abstractmethod
    def forward(self, input):
        pass


    @abstractmethod
    def backward(self, error, y):
        pass


    @abstractmethod
    def outputShape(self):
        pass


    @abstractmethod
    def updateGradient(self):
        pass
