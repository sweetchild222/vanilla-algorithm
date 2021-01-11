from model.layer.abs_layer import *


class Input(ABSLayer):

    def __init__(self, input_shape, backward_layer=None):
        super(Input, self).__init__(backward_layer)
        self.input_shape = input_shape

    def start(self):
        pass

    def forward(self, input):
        return input

    def backward(self, error, y):
        return error

    def outputShape(self):
        return self.input_shape

    def updateGradient(self):
        pass
