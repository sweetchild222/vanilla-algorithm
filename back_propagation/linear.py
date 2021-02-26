import numpy as np


def linear_forward(x, weight, bias):

    return np.dot(x, weight) + bias

def linear_backward(error, weight):

    return np.dot(error, weight.T)


def train(X, T, learning_rate, iterate):

    inputNodes = X.shape[-1]
    outputNodes = T.shape[-1]

    weight = np.full((inputNodes, outputNodes), 1.0)
    bias = np.zeros((outputNodes))

    for i in range(iterate):

        y = linear_forward(X, weight, bias)

        g = np.average((y - T)**2)
        print(i, ' mse : ', g)
        error = (y - T)

        weight_delta = np.dot(X.T, error)
        bias_delta = np.sum(error, axis=0)

        error = linear_backward(error, weight)

        weight -= (learning_rate * weight_delta)
        bias -= (learning_rate * bias_delta)

    return weight, bias

X = np.array([[2.0]])
T = np.array([[11.0]])

weight, bias = train(X, T, learning_rate = 0.1, iterate = 5)

y1 = linear_forward(X, weight, bias)

print('test : ', y1)
