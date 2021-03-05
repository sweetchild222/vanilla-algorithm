import numpy as np

from cnn.loader import *
import core.convolution as conv
import core.pooling as pool
import core.sigmoid as sigmoid


def print_summary(epoch, iterate, mse, predict):

    if ((epoch + 1) % 10) == 0:
        print('epoch : ', (epoch + 1), '    mse : ', mse)

    if (iterate - epoch) == 1:
        np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
        print('predict', predict)


def shuffle(x, target):

    shuffle_indices = np.arange(x.shape[0])
    np.random.shuffle(shuffle_indices)

    return x[shuffle_indices], target[shuffle_indices]


def train(x, target, convol_size, convol_stride, pool_size, pool_stride, layer_count, learning_rate, iterate):

    weight_list = [np.random.normal(size=convol_size).copy()] * layer_count
    bias_list = [np.zeros((1)).copy()] * layer_count

    for i in range(iterate):

        x, target = shuffle(x, target)

        input = x

        input_conv_list = []
        input_sigmoid_list = []
        input_pool_list = []

        for l in range(layer_count):
            input_conv_list.append(input)
            c = conv.forward(input, weight_list[l], bias_list[l], convol_stride)

            input_sigmoid_list.append(c)
            s = sigmoid.forward(c)

            input_pool_list.append(s)
            p = pool.forward(s, pool_size, pool_stride)

            input = p

        predict = p

        error = (predict - target)

        mse = np.average(error**2)

        print_summary(i, iterate, mse, predict)

        for l in reversed(range(layer_count)):

            input_pool = input_pool_list[l]
            error = pool.backward(input_pool, error, pool_size, pool_stride)

            input_sigmoid = input_sigmoid_list[l]
            error = sigmoid.backward(input_sigmoid, error)

            weight = weight_list[l]
            bias = bias_list[l]

            input_conv = input_conv_list[l]
            error, weight_delta, bias_delta = conv.backward(input_conv, error, weight, bias, convol_stride)

            weight -= (learning_rate * weight_delta)
            bias -= (learning_rate * bias_delta)

    return weight_list, bias_list


convol_size = (3, 3)
convol_stride = (1, 1)
pool_size = (2, 2)
pool_stride = pool_size

train_x, train_t, test_x, test_t = loadDataSet('cnn/image/emotion/train', 'cnn/image/emotion/target', 'cnn/image/emotion/test')

layer_count = 3

for l in range(layer_count):
    train_t = pool.forward(train_t, pool_size, pool_stride)
    test_t = pool.forward(test_t, pool_size, pool_stride)

print(train_t)

weight_list, bias_list = train(train_x, train_t, convol_size, convol_stride, pool_size, pool_stride, layer_count, learning_rate = 0.001, iterate = 5000)

input = test_x

for l in range(layer_count):
    c = conv.forward(input, weight_list[l], bias_list[l], convol_stride)
    s = sigmoid.forward(c)
    p = pool.forward(s, pool_size, pool_stride)
    input = p

predict = p

predict = np.where(predict >= 0.5, 1.0, predict)
predict = np.where(predict < 0.5, 0.0, predict)

print('predict', predict)
print('target', test_t)

equal = np.array_equal(predict, test_t)
print('equal : ', equal)
