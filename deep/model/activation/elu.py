import numpy as np
from abc import *
from model.activation.abs_activation import *

class ELU(ABSActivation):

    def __init__(self, alpha):
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, output):
        output = np.where(output > 0, output, (np.exp(output) - 1) * self.alpha)
        self.last_output = output
        return output

    def backward(self, error, y):
        return error * np.where(self.last_output > 0, 1, np.exp(error) * self.alpha)
