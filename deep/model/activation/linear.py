import numpy as np
from abc import *
from model.activation.abs_activation import *

class Linear(ABSActivation):

    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, output):
        return output

    def backward(self, error, y):
        return error
