import numpy as np

import core.activation as act
import core.linear as linear


def print_summary(epoch, mse):

    if ((epoch + 1) % 100) == 0:
        print('epoch : ', (epoch + 1), '    mse : ', mse)


def train(x, target, learning_rate, iterate):

    inputNodes = x.shape[-1]
    outputNodes = target.shape[-1]

    weight = np.zeros((inputNodes, outputNodes))
    bias = np.zeros((outputNodes))

    for i in range(iterate):

        y = linear.forward(x, weight, bias)
        predict = act.sigmoid_forward(y)

        error = (predict - target)

        mse = np.average(error**2)

        print_summary(i, mse)

        error = act.sigmoid_backward(y, error)

        error, weight_delta, bias_delta = linear.backward(x, error, weight, bias)

        weight -= (learning_rate * weight_delta)
        bias -= (learning_rate * bias_delta)

    return weight, bias

train_x = np.array([[5]])
target = np.array([[1]])

weight, bias = train(train_x, target, learning_rate = 0.2, iterate = 500)

y = linear.forward(train_x, weight, bias)
predict = act.sigmoid_forward(y)

print('predict : ', predict)
