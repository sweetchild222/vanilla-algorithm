import numpy as np

def LinearRegression(X, T, iteration):

    weight = np.zeros(X.shape[-1])
    bias = np.zeros(1)
    learning_rate = 1e-4

    for i in range(iteration):
        error = prediction(weight, bias, X) - T

        weight_delta, bias_delta = gradient(X, error, learning_rate)
        weight -= weight_delta
        bias -= bias_delta

    return weight, bias


def prediction(weight, bias, X):

    return np.dot(X, weight) + bias


def gradient(X, error, learning_rate):

    weight_delta = (learning_rate * (1 / len(error)) * (np.dot(error, X)))
    bias_delta = (learning_rate * 100 * len(error) * np.sum(error))

    return weight_delta, bias_delta


X = np.array([[5.0, 50], [6.0, 20], [10.0, 30], [7.0, 40], [8.0, 20], [12.0, 60]])
T = np.array([13.0, 15.5, 22.5, 17.0, 20.0, 26.5])
iteration = 100000

weight, bias  = LinearRegression(X, T, iteration)

print('weight  : ', weight)
print('bias  : ', bias)

X_test = np.array([[9.0, 40]])

predicted_Y = prediction(weight, bias, X_test)

print('X : ', X_test, ', predicted Y : ', predicted_Y)
