import numpy as np
import matplotlib.pyplot as plt
from loader import *


def activation_forward(x):

    return 1 / (1 + np.exp(-x))


def padding(input, weight_shape):

    (weight_height, weight_width) = weight_shape

    pad_height = weight_height // 2
    pad_width = weight_width // 2

    return np.pad(input, ((0, 0), (pad_height, pad_height), (pad_width, pad_width)), 'constant', constant_values=0)


def convolution_forward(input, weight, bias, stride):

    input = padding(input, weight.shape)

    (batches, input_height, input_width) = input.shape
    (weight_height, weight_width) = weight.shape
    (stride_y, stride_x) = stride

    numerator_height = input_height - weight_height
    numerator_width = input_width - weight_width

    output_shape = ((numerator_height // stride_y) + 1, (numerator_width // stride_x) + 1)

    output = np.zeros((batches, ) + output_shape)

    input_y = out_y = 0
    while (input_y + weight_height) <= input_height:
        input_x = out_x = 0
        while (input_x + weight_width) <= input_width:

            i = input[:,input_y:input_y + weight_height, input_x:input_x + weight_width]

            iw = i * weight

            iw = iw.reshape((iw.shape[0], -1))
            iw = np.sum(iw, axis=-1)

            output[:, out_y, out_x] =  iw + bias

            input_x += stride_x
            out_x += 1

        input_y += stride_y
        out_y += 1

    return output


def activation_backward(input, error):

    return error * input * (1.0 - input)


def convolution_backward(input, error, weight, bias, stride):

    input = padding(input, weight.shape)

    (weight_height, weight_width) = weight.shape
    (batches, input_height, input_width) = input.shape
    (stride_y, stride_x) = stride

    back_layer_error = np.zeros(input.shape)
    batch_weight = np.array([weight] * batches)

    w_delta = np.zeros((batches, ) + weight.shape)
    b_delta = np.zeros((batches, ) + bias.shape)

    input_y = out_y = 0
    while (input_y + weight_height) <= input_height:
        input_x = out_x = 0
        while (input_x + weight_width) <= input_width:
            err = (error[:, out_y, out_x])[:, np.newaxis, np.newaxis]
            i = input[:, input_y:input_y + weight_height, input_x:input_x + weight_width]

            w_delta += (err * i)
            b_delta += err.reshape((batches, ) + bias.shape)

            bw_err = batch_weight * err
            #shallow copy
            bl_err = back_layer_error[:, input_y:input_y + weight_height, input_x:input_x + weight_width]
            bl_err += bw_err

            input_x += stride_x
            out_x += 1


        input_y += stride_y
        out_y += 1

    return back_layer_error, np.sum(w_delta, axis=0), np.sum(b_delta, axis=0)


def pooling_forward(input, pool_size, pool_stride):

    (batches, input_height, input_width) = input.shape
    (pool_height, pool_width) = pool_size
    (stride_y, stride_x) = pool_stride

    output_shape = ((input_height // stride_y), (input_width // stride_x))

    output = np.zeros((batches, ) + output_shape)

    input_y = out_y = 0
    while (input_y + pool_height) <= input_height:
        input_x = out_x = 0
        while (input_x + pool_width) <= input_width:
            i = input[:, input_y:input_y + pool_height, input_x:input_x + pool_width]
            max_pool = np.max(i.reshape((batches, -1)), axis=-1)
            output[:, out_y, out_x] = max_pool

            input_x += stride_x
            out_x += 1

        input_y += stride_y
        out_y += 1

    return output


def train(X, T, convol_size, convol_stride, pool_size, pool_stride, learning_rate, iterate):

    weight = np.random.normal(size=convol_size)
    bias = np.zeros((1))

    for i in range(iterate):

        y = convolution_forward(X, weight, bias, convol_stride)

        s = activation_forward(y)

        g = np.average((s - T)**2)

        if ((i + 1) % 1000) == 0:
            print('epoch : ', (i + 1), '    mse : ', g)
            if (iterate - i) == 1:
                np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})
                print('convolution output', y)
                print('activation output', s)
                p = pooling_forward(s, pool_size, pool_stride)
                print('pooling output', p)


        error = (s - T)

        error = activation_backward(s, error)

        error, weight_delta, bias_delta = convolution_backward(X, error, weight, bias, convol_stride)

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

y = convolution_forward(test_x, weight, bias, convol_stride)
predict = activation_forward(y)

predict = np.where(predict >= 0.5, 1.0, predict)
predict = np.where(predict < 0.5, 0.0, predict)

print('predict', predict)

equal = np.array_equal(predict, test_t)
print('equal : ', equal)