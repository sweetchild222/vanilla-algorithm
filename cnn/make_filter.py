import numpy as np
import matplotlib.pyplot as plt
from loader import *

import core.convolution as conv
import core.pooling as pool
import core.activation as act



def train(X, T, convol_size, convol_stride, pool_size, pool_stride, learning_rate, iterate):

    weight = np.random.normal(size=convol_size)
    bias = np.zeros((1))

    for i in range(iterate):

        y = conv.forward(X, weight, bias, convol_stride)

        s = act.sigmoid_forward(y)

        g = np.average((s - T)**2)

        if ((i + 1) % 1000) == 0:
            print('epoch : ', (i + 1), '    mse : ', g)
            if (iterate - i) == 1:
                np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})
                print('convolution output', y)
                print('activation output', s)
                p = pool.forward(s, pool_size, pool_stride)
                print('pooling output', p)


        error = (s - T)

        error = act.sigmoid_backward(s, error)

        error, weight_delta, bias_delta = conv.backward(X, error, weight, bias, convol_stride)

        weight -= (learning_rate * weight_delta)
        bias -= (learning_rate * bias_delta)

    return weight, bias


convol_size = (3, 3)
convol_stride = (1, 1)
pool_size = (2, 2)
pool_stride = pool_size

train_x, train_t, test_x, test_t = loadDataSet('image/shape/train', 'image/shape/target', 'image/shape/test')

weight, bias = train(train_x, train_t, convol_size, convol_stride, pool_size, pool_stride, learning_rate = 0.001, iterate = 30000)

print('weight: ', weight)
print('bias: ', bias)

y = conv.forward(test_x, weight, bias, convol_stride)
predict = act.sigmoid_forward(y)

predict = np.where(predict >= 0.5, 1.0, predict)
predict = np.where(predict < 0.5, 0.0, predict)

print('predict', predict)

equal = np.array_equal(predict, test_t)
print('equal : ', equal)
