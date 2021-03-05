import numpy as np


def forward(input):

    return 1 / (1 + np.exp(-input))


def backward(input, error):

    input = forward(input)

    return error * input * (1.0 - input)


#def activation_backward(error, input):

    #input = activation_forward(input)

    #return error * input * (1.0 - input)
