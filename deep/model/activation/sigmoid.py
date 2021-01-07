import numpy as np
from abc import *
from model.activation.abs_activation import *

class Sigmoid(ABSActivation):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, output):
        output = 1 / (1 + np.exp(-output))
        self.last_output = output
        return output

    def backward(self, error, y):
        return error * self.last_output * (1.0 - self.last_output)
