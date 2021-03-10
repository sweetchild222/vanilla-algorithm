import numpy as np


def print_summary(epoch, mse):

    if ((epoch + 1) % 10000) == 0:
        print('epoch : ', (epoch + 1), '    mse : ', mse)


def train(x, target, learning_rate, epochs):

    inputNodes = x.shape[-1]
    outputNodes = target.shape[-1]
    weight = np.zeros((inputNodes, outputNodes))
    bias = np.zeros(outputNodes)

    for i in range(epochs):
        y = test(x, weight, bias)

        error = y - target

        mse = np.average(error**2)

        print_summary(i, mse)

        weight_delta, bias_delta = gradient(x, error)

        weight -= (learning_rate * weight_delta)
        bias -= (learning_rate * bias_delta)

    return weight, bias


def test(x, weight, bias):

    return np.dot(x, weight) + bias


def gradient(x, error):

    weight_delta = np.dot(x.T, error)
    bias_delta = np.sum(error, axis=0)

    return weight_delta, bias_delta


x = np.array([[5.0, 50], [6.0, 20], [10.0, 30], [7.0, 40], [8.0, 20], [12.0, 60]])
target = np.array([[13.0], [15.5], [22.5], [17.0], [20.0], [26.5]])

weight, bias = train(x, target, learning_rate = 1e-4, epochs = 100000)

print('weight  : ', weight)
print('bias  : ', bias)

test_x = np.array([[9.0, 40]])

y = test(test_x, weight, bias)

print('test x : ', test_x)
print('predict y : ', y)
