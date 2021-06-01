import numpy as np

from cnn.loader import *
import core.convolution as conv
import core.pooling as pool
import core.sigmoid as sigmoid


def print_summary(epoch, remain, mse, predict, target, convolution):

    if ((epoch + 1) % 1000) == 0:
        print_mse(epoch, mse)

    if remain == 1:
        print_output(convolution, predict)


def print_output(convolution, activation):

    np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})

    print('convolution output', convolution)
    print('activation output', activation)

def print_mse(epoch, mse):
    print('epoch : ', (epoch + 1), '    mse : ', mse)


def train(x, target, convol_size, convol_stride, pool_size, pool_stride, learning_rate, iterate):

    weight = np.random.normal(size=convol_size)
    bias = np.zeros((1))

    for i in range(iterate):

        y = conv.forward(x, weight, bias, convol_stride)

        predict = sigmoid.forward(y)

        error = (predict - target)

        mse = np.average(error**2)

        print_summary(i, (iterate - i), mse, predict, target, y)

        error = sigmoid.backward(y, error)

        error, weight_delta, bias_delta = conv.backward(x, error, weight, bias, convol_stride)

        weight -= (learning_rate * weight_delta)
        bias -= (learning_rate * bias_delta)

    return weight, bias


convol_size = (3, 3)
convol_stride = (1, 1)
pool_size = (2, 2)
pool_stride = pool_size

train_x, train_t, test_x, test_t = loadDataSet('cnn/image/line/train', 'cnn/image/line/target', 'cnn/image/line/test')
#train_x, train_t, test_x, test_t = loadDataSet('cnn/image/shape/train', 'cnn/image/shape/target', 'cnn/image/shape/test')


weight, bias = train(train_x, train_t, convol_size, convol_stride, pool_size, pool_stride, learning_rate = 0.001, iterate = 30000)

print('weight: ', weight)
print('bias: ', bias)

y = conv.forward(test_x, weight, bias, convol_stride)
predict = sigmoid.forward(y)

predict = np.where(predict >= 0.5, 1.0, predict)
predict = np.where(predict < 0.5, 0.0, predict)

print('predict', predict)

equal = np.array_equal(predict, test_t)
print('equal : ', equal)
