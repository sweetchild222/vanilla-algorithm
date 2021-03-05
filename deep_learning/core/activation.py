import numpy as np


def sigmoid_forward(input):

    return 1 / (1 + np.exp(-input))


def sigmoid_backward(input, error):

    input = sigmoid_forward(input)

    return error * input * (1.0 - input)


#def activation_backward(error, input):

    #input = activation_forward(input)

    #return error * input * (1.0 - input)
