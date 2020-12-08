import numpy as np
from layer.abs_layer import *
import operator
from functools import reduce


class Flatten(ABSLayer):

    def __init__(self, backward_layer):
        super(Flatten, self).__init__(backward_layer)

    def forward(self, input):
        return input.reshape((-1, 1))

    def backward(self, error, y):
        return error.reshape(self.input_shape)

    def outputShape(self):
        return (reduce(operator.mul, self.input_shape), )

    def updateGradient(self):
        pass
