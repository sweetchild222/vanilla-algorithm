import numpy as np
from abc import *
from model.activation.abs_activation import *

class Softmax(ABSActivation):

    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, output):
        output = np.exp(output)
        return output / np.sum(output)

    def backward(self, error, y):
        return (error - y)
