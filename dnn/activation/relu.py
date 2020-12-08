from abc import *
from activation.abs_activation import *

class Relu(ABSActivation):
    
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, output):
        output[output < 0] = 0
        self.last_output = output
        return output

    def backward(self, error, y):
        error[self.last_output < 0] = 0
        return error
