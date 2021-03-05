import numpy as np

from cnn.loader import *
import core.convolution as conv
import core.pooling as pool
import core.sigmoid as sigmoid


def train(x, target, convol_size, convol_stride, pool_size, pool_stride, learning_rate, iterate):

    c1_weight = np.random.normal(size=convol_size)
    c1_bias = np.zeros((1))

    for i in range(iterate):

        c1 = conv.forward(x, c1_weight, c1_bias, convol_stride)

        cs1 = sigmoid.forward(c1)

        predict = pool.forward(cs1, pool_size, pool_stride)

        error = (predict - target)

        mse = np.average(error**2)

        if ((i + 1) % 1000) == 0:
            print('epoch : ', i + 1, '    mse : ', g)
            if (iterate - i) == 1:
                np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
                print('output', p)


        error = pool.backward(cs1, error, pool_size, pool_stride)

        error = sigmoid.backward(c1, error)

        error, weight_delta, bias_delta = conv.backward(x, error, c1_weight, c1_bias, convol_stride)

        c1_weight -= (learning_rate * weight_delta)
        c1_bias -= (learning_rate * bias_delta)

    return c1_weight, c1_bias


convol_size = (3, 3)
convol_stride = (1, 1)
pool_size = (2, 2)
pool_stride = pool_size

train_x, train_t, test_x, test_t = loadDataSet('cnn/image/emotion/train', 'cnn/image/emotion/target', 'cnn/image/emotion/test')

train_t = pool.forward(train_t, pool_size, pool_stride)
test_t = pool.forward(test_t, pool_size, pool_stride)

print(train_t)

c1_weight, c1_bias = train(train_x, train_t, convol_size, convol_stride, pool_size, pool_stride, learning_rate = 0.001, iterate = 100)

y = conv.forward(test_x, c1_weight, c1_bias, convol_stride)
s = sigmoid.forward(y)
predict = pool.forward(s, pool_size, pool_stride)

predict = np.where(predict >= 0.5, 1.0, predict)
predict = np.where(predict < 0.5, 0.0, predict)

print('predict', predict)
print('test', test_t)

equal = np.array_equal(predict, test_t)
print('equal : ', equal)