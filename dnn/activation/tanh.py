import numpy as np
from abc import *
from activation.abs_activation import *

class Tanh(ABSActivation):

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, output):
        output = np.tanh(output)
        return output

    def backward(self, error, y):
        return 1 - (np.tanh(error))**2
