import numpy as np
import matplotlib.pyplot as plt


def activation_forward(x):

    return 1 / (1 + np.exp(-x))


def linear_forward(x, weight, bias):

    return np.dot(x, weight) + bias


def activation_backward(error, input):

    return error * input * (1.0 - input)


def linear_backward(error, weight):

    return np.dot(error, weight.T)


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

        error = (s2 - T)

        error = activation_backward(error, s2)

        h2_weight_delta = np.dot(s1.T, error)
        h2_bias_delta = np.sum(error, axis=0)

        error = linear_backward(error, h2_weight_delta)

        h2_weight -= (learning_rate * h2_weight_delta)
        h2_bias -= (learning_rate * h2_bias_delta)

        error = activation_backward(error, s1)

        h1_weight_delta = np.dot(X.T, error)
        h1_bias_delta = np.sum(error, axis=0)

        error = linear_backward(error, h1_weight_delta)

        h1_weight -= (learning_rate * h1_weight_delta)
        h1_bias -= (learning_rate * h1_bias_delta)

    return h1_weight, h1_bias, h2_weight, h2_bias

X = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0],
              [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1],
              [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2],
              [0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3],
              [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4],
              [0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5],
              [0, 6], [1, 6], [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [7, 6],
              [0, 7], [1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [6, 7], [7, 7]])

T = np.array([[0], [0], [0], [0], [0], [0], [0], [0],
              [0], [1], [1], [0], [0], [1], [1], [0],
              [1], [1], [1], [1], [1], [1], [1], [1],
              [1], [1], [1], [1], [1], [1], [1], [1],
              [0], [1], [1], [1], [1], [1], [1], [0],
              [0], [0], [1], [1], [1], [1], [0], [0],
              [0], [0], [0], [1], [1], [0], [0], [0],
              [0], [0], [0], [0], [0], [0], [0], [0]])

h1_weight, h1_bias, h2_weight, h2_bias = train(X, T, learning_rate = 0.01, iterate = 5000000)

#print('h1_weight  : ', h1_weight)
#print('h1_bias  : ', h1_bias)
#print('h2_weight  : ', h2_weight)
#print('h2_bias  : ', h2_bias)

y1 = linear_forward(X, h1_weight, h1_bias)
s1 = activation_forward(y1)

y2 = linear_forward(s1, h2_weight, h2_bias)
s2 = activation_forward(y2)

print(s2)

s2 = np.where(s2 >= 0.5, 1, 0)



#s2[0.5 >= s2] = 1
#s2[0.5 < s2] = 0

print(s2.reshape((8,8)))
