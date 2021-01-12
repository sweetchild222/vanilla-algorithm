from model.layer.abs_layer import *


class Input(ABSLayer):

    def __init__(self, input_shape, backward_layer=None):
        super(Input, self).__init__(backward_layer)
        self.input_shape = input_shape


    def forward(self, input):
        return input


    def backward(self, error, y):
        return error


    def beginBatch(self):
        pass


    def endBatch(self):
        pass


    def outputShape(self):
        return self.input_shape
