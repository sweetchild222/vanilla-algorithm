import numpy as np
import core.sigmoid as sigmoid
import core.linear as linear


def print_summary(epoch, mse):

    if ((epoch + 1) % 1000) == 0:
        print('epoch : ', (epoch + 1), '    mse : ', mse)


def train(x, target, learning_rate, iterate):

    inputNodes = x.shape[-1]
    outputNodes = target.shape[-1]

    h1_units = 25

    h1_weight = np.random.normal(size=(inputNodes, h1_units))
    h1_bias = np.zeros((h1_units))

    h2_weight = np.random.normal(size=(h1_units, outputNodes))
    h2_bias = np.zeros((outputNodes))

    for i in range(iterate):

        y1 = linear.forward(x, h1_weight, h1_bias)
        s1 = sigmoid.forward(y1)
        y2 = linear.forward(s1, h2_weight, h2_bias)
        predict = sigmoid.forward(y2)

        error = (predict - target)

        mse = np.average(error**2)

        print_summary(i, mse)

        error = sigmoid.backward(y2, error)

        error, h2_weight_delta, h2_bias_delta = linear.backward(s1, error, h2_weight, h2_bias)

        h2_weight -= (learning_rate * h2_weight_delta)
        h2_bias -= (learning_rate * h2_bias_delta)

        error = sigmoid.backward(y1, error)

        error, h1_weight_delta, h1_bias_delta = linear.backward(x, error, h1_weight, h1_bias)

        h1_weight -= (learning_rate * h1_weight_delta)
        h1_bias -= (learning_rate * h1_bias_delta)

    return h1_weight, h1_bias, h2_weight, h2_bias

train_x = np.array([[0, 0], [1, 1]])
target = np.array([[0], [1]])

h1_weight, h1_bias, h2_weight, h2_bias = train(train_x, target, learning_rate = 0.01, iterate = 5000)

y1 = linear.forward(train_x, h1_weight, h1_bias)
s1 = sigmoid.forward(y1)

y2 = linear.forward(s1, h2_weight, h2_bias)
predict = sigmoid.forward(y2)

print('predict : ', predict)
