import numpy as np


def train(X, T, learning_rate, iteration):

    inputNodes = X.shape[-1]
    outputNodes = T.shape[-1]
    weight = np.zeros((inputNodes, outputNodes))
    bias = np.zeros(outputNodes)

    for i in range(iteration):
        error = test(X, weight, bias) - T

        weight_delta, bias_delta = gradient(X, error, learning_rate)

        weight -= weight_delta
        bias -= bias_delta

    return weight, bias


def test(X, weight, bias):

    return np.dot(X, weight) + bias


def gradient(X, error, learning_rate):

    lr = learning_rate

    weight_delta = lr * np.dot(X.T, error)
    bias_delta = lr * np.sum(error, axis=0)

    return weight_delta, bias_delta


X = np.array([[5.0, 50], [6.0, 20], [10.0, 30], [7.0, 40], [8.0, 20], [12.0, 60]])
T = np.array([[13.0], [15.5], [22.5], [17.0], [20.0], [26.5]])

weight, bias = train(X, T, learning_rate = 1e-4, iteration = 100000)

print('weight  : ', weight)
print('bias  : ', bias)

X_test = np.array([[9.0, 40]])

predicted_Y = test(X_test, weight, bias)

print('X : ', X_test, ', predicted Y : ', predicted_Y)
