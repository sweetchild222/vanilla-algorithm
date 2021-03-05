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


def train(x, target, convol_size, convol_stride, pool_size, pool_stride, learning_rate, iterate):

    c1_weight = np.random.normal(size=convol_size)
    c1_bias = np.zeros((1))

    c2_weight = np.random.normal(size=convol_size)
    c2_bias = np.zeros((1))

    c3_weight = np.random.normal(size=convol_size)
    c3_bias = np.zeros((1))

    for i in range(iterate):

        x, target = shuffle(x, target)

        c1 = conv.forward(x, c1_weight, c1_bias, convol_stride)

        s1 = sigmoid.forward(c1)

        p1 = pool.forward(s1, pool_size, pool_stride)

        c2 = conv.forward(p1, c2_weight, c2_bias, convol_stride)

        s2 = sigmoid.forward(c2)

        p2 = pool.forward(s2, pool_size, pool_stride)

        c3 = conv.forward(p2, c3_weight, c3_bias, convol_stride)

        s3 = sigmoid.forward(c3)

        predict = pool.forward(s3, pool_size, pool_stride)

        error = (predict - target)

        mse = np.average(error**2)

        print_summary(i, iterate, mse, predict)


        error = pool.backward(s3, error, pool_size, pool_stride)

        error = sigmoid.backward(c3, error)

        error, weight_delta, bias_delta = conv.backward(p2, error, c3_weight, c3_bias, convol_stride)

        c3_weight -= (learning_rate * weight_delta)
        c3_bias -= (learning_rate * bias_delta)

        error = pool.backward(s2, error, pool_size, pool_stride)

        error = sigmoid.backward(c2, error)

        error, weight_delta, bias_delta = conv.backward(p1, error, c2_weight, c2_bias, convol_stride)

        c2_weight -= (learning_rate * weight_delta)
        c2_bias -= (learning_rate * bias_delta)

        error = pool.backward(s1, error, pool_size, pool_stride)

        error = sigmoid.backward(c1, error)

        error, weight_delta, bias_delta = conv.backward(x, error, c1_weight, c1_bias, convol_stride)

        c1_weight -= (learning_rate * weight_delta)
        c1_bias -= (learning_rate * bias_delta)

    return c1_weight, c1_bias, c2_weight, c2_bias, c3_weight, c3_bias


convol_size = (3, 3)
convol_stride = (1, 1)
pool_size = (2, 2)
pool_stride = pool_size

train_x, train_t, test_x, test_t = loadDataSet('cnn/image/emotion/train', 'cnn/image/emotion/target', 'cnn/image/emotion/test')

train_t = pool.forward(train_t, pool_size, pool_stride)
train_t = pool.forward(train_t, pool_size, pool_stride)
train_t = pool.forward(train_t, pool_size, pool_stride)
test_t = pool.forward(test_t, pool_size, pool_stride)
test_t = pool.forward(test_t, pool_size, pool_stride)
test_t = pool.forward(test_t, pool_size, pool_stride)

print(train_t)

c1_weight, c1_bias, c2_weight, c2_bias, c3_weight, c3_bias = train(train_x, train_t, convol_size, convol_stride, pool_size, pool_stride, learning_rate = 0.001, iterate = 5000)

c1 = conv.forward(test_x, c1_weight, c1_bias, convol_stride)
s1 = sigmoid.forward(c1)
p1 = pool.forward(s1, pool_size, pool_stride)

c2 = conv.forward(p1, c2_weight, c2_bias, convol_stride)
s2 = sigmoid.forward(c2)
p2 = pool.forward(s2, pool_size, pool_stride)

c3 = conv.forward(p2, c3_weight, c3_bias, convol_stride)
s3 = sigmoid.forward(c3)
predict = pool.forward(s3, pool_size, pool_stride)


predict = np.where(predict >= 0.5, 1.0, predict)
predict = np.where(predict < 0.5, 0.0, predict)

print('predict', predict)
print('target', test_t)

equal = np.array_equal(predict, test_t)
print('equal : ', equal)
