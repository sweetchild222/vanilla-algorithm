import numpy as np
from abc import *
from activation.abs_activation import *

class LeakyRelu(ABSActivation):

    def __init__(self, alpha):
        super(LeakyRelu, self).__init__()
        self.alpha = alpha

    def forward(self, output):
        output = np.where(output > 0, output, output * self.alpha)
        self.last_output = output
        return output

    def backward(self, error, y):
        return error * np.where(self.last_output > 0, 1, self.alpha)
