import numpy as np

import core.linear as linear


def print_summary(epoch, mse):

    if ((epoch + 1) % 1) == 0:
        print('epoch : ', (epoch + 1), '    mse : ', mse)


def train(x, target, learning_rate, iterate):

    inputNodes = x.shape[-1]
    outputNodes = target.shape[-1]

    weight = np.full((inputNodes, outputNodes), 1.0)
    bias = np.zeros((outputNodes))

    for i in range(iterate):

        predict = linear.forward(x, weight, bias)

        error = (predict - target)

        mse = np.average(error**2)

        print_summary(i, mse)

        error, weight_delta, bias_delta = linear.backward(x, error, weight, bias)

        weight -= (learning_rate * weight_delta)
        bias -= (learning_rate * bias_delta)

    return weight, bias

train_x = np.array([[2.0]])
target = np.array([[11.0]])

weight, bias = train(train_x, target, learning_rate = 0.1, iterate = 5)

predict = linear.forward(train_x, weight, bias)

print('predict : ', predict)
