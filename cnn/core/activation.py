import numpy as np


def sigmoid_forward(x):

    return 1 / (1 + np.exp(-x))


def sigmoid_backward(input, error):

    return error * input * (1.0 - input)
