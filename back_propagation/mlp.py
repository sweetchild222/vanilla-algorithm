import numpy as np
import matplotlib.pyplot as plt


def activation_forward(x):

    return 1 / (1 + np.exp(-x))


def linear_forward(x, weight, bias):

    return np.dot(x, weight) + bias


def activation_backward(error, input):

    return error * input * (1.0 - input)


def linear_backward(input, error, weight, bias):

    weight_delta = np.dot(input.T, error)
    bias_delta = np.sum(error, axis=0)
    back_layer_error = np.dot(error, weight.T)

    return back_layer_error, weight_delta, bias_delta


def train(X, T, learning_rate, iterate):

    inputNodes = X.shape[-1]
    outputNodes = T.shape[-1]

    h1_units = 25

    h1_weight = np.random.normal(size=(inputNodes, h1_units))
    h1_bias = np.zeros((h1_units))

    h2_weight = np.random.normal(size=(h1_units, outputNodes))
    h2_bias = np.zeros((outputNodes))

    for i in range(iterate):

        y1 = linear_forward(X, h1_weight, h1_bias)
        s1 = activation_forward(y1)
        y2 = linear_forward(s1, h2_weight, h2_bias)
        s2 = activation_forward(y2)

        g = np.average((s2 - T)**2)

        if ((i + 1) % 1000) == 0:
            print('epoch : ', (i + 1), '    mse : ', g)

        error = (s2 - T)

        error = activation_backward(error, s2)

        error, h2_weight_delta, h2_bias_delta = linear_backward(s1, error, h2_weight, h2_bias)

        h2_weight -= (learning_rate * h2_weight_delta)
        h2_bias -= (learning_rate * h2_bias_delta)

        error = activation_backward(error, s1)

        error, h1_weight_delta, h1_bias_delta = linear_backward(X, error, h1_weight, h1_bias)

        h1_weight -= (learning_rate * h1_weight_delta)
        h1_bias -= (learning_rate * h1_bias_delta)

    return h1_weight, h1_bias, h2_weight, h2_bias

X = np.array([[0, 0], [1, 1]])
T = np.array([[0], [1]])

h1_weight, h1_bias, h2_weight, h2_bias = train(X, T, learning_rate = 0.01, iterate = 5000)

y1 = linear_forward(X, h1_weight, h1_bias)
s1 = activation_forward(y1)

y2 = linear_forward(s1, h2_weight, h2_bias)
s2 = activation_forward(y2)

print('test : ', s2)
