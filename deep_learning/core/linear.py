import numpy as np

def forward(input, weight, bias):

    return np.dot(input, weight) + bias


def backward(input, error, weight, bias):

    weight_delta = np.dot(input.T, error)
    bias_delta = np.sum(error, axis=0)
    back_layer_error = np.dot(error, weight.T)

    return back_layer_error, weight_delta, bias_delta
