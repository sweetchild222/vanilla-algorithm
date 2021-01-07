import numpy as np
from abc import *
from model.activation.abs_activation import *

class Tanh(ABSActivation):

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, output):
        output = np.tanh(output)
        self.last_output = output
        return output

    def backward(self, error, y):
        return error * (1 - (np.tanh(self.last_output))**2)
