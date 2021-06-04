import numpy as np

import core.sigmoid as sigmoid
import core.linear as linear


def print_summary(epoch, loss):

    #if ((epoch + 1) % 100) == 0:
    print('epoch : ', (epoch + 1), '    loss : ', loss)


def print_numpy(key, value):
    print(key, ' : ', np.round_(value, 3))


def train(x, target, learning_rate, iterate):

    inputNodes = x.shape[-1]
    outputNodes = target.shape[-1]

    weight = np.zeros((inputNodes, outputNodes))
    bias = np.zeros((outputNodes))

    for i in range(iterate):

        y = linear.forward(x, weight, bias)
        p = sigmoid.forward(y)

        p_error = (p - target)

        loss = np.average(p_error**2)
        print_numpy('w', weight)
        print_numpy('b', bias)
        print_numpy('y', y)
        print_numpy('p', p)
        print_numpy('L', loss)

        y_error = sigmoid.backward(y, p_error)
        x_error, w_error, b_error = linear.backward(x, y_error, weight, bias)

        weight -= (learning_rate * w_error)
        bias -= (learning_rate * b_error)

        print_numpy('~p', p_error)
        print_numpy('~y', y_error)
        print_numpy('~x', x_error)
        print_numpy('~w', w_error)
        print_numpy('~b', b_error)

        print('----- epoch : ', (i + 1), '------')

    return weight, bias

train_x = np.array([[5]])
target = np.array([[1]])

weight, bias = train(train_x, target, learning_rate = 0.2, iterate = 4)

y = linear.forward(train_x, weight, bias)
predict = sigmoid.forward(y)

print('predict : ', predict)
