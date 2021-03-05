import numpy as np


def sigmoid_forward(input):

    return 1 / (1 + np.exp(-input))


def sigmoid_backward(input, error):

    output = sigmoid_forward(input)

    return error * output * (1.0 - output)
