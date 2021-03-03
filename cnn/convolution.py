import numpy as np
import matplotlib.pyplot as plt
from loader import *


def activation_forward(x):

    return 1 / (1 + np.exp(-x))

def padding(x):

    pad_size = 1
    return np.pad(x, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=0)

def convolution_forward(input, weight, bias):

    (batches, input_height, input_width) = input.shape
    (kernel_height, kernel_width) = weight.shape
    (stride_y, stride_x) = (1, 1)

    numerator_height = input_height - kernel_height
    numerator_width = input_width - kernel_width

    calc_shape = ((numerator_height // stride_y) + 1, (numerator_width // stride_x) + 1)
    #a = np.array([[[1,1,1], [1,1,1], [1,1,1]], [[2,2,2], [2,2,2], [2,2,2]], [[3,3,3], [3,3,3], [3,3,3]], [[4,4,4], [4,4,4], [4,4,4]]])
    #b = np.array([[2,2,2], [2,2,2], [2,2,2]])

    output = np.zeros((batches, ) + calc_shape)

    input_y = out_y = 0
    while (input_y + kernel_height) <= input_height:
        input_x = out_x = 0
        while (input_x + kernel_width) <= input_width:

            i = input[:,input_y:input_y + kernel_height, input_x:input_x + kernel_width]

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


def convolution_backward(input, error, weight, bias):

    (batches, input_height, input_width) = input.shape
    (kernel_height, kernel_width) = weight.shape
    (stride_y, stride_x) = (1, 1)

    back_layer_error = np.zeros(input.shape)
    batch_weight = np.array([weight] * batches)

    w_delta = np.zeros((batches, ) + weight.shape)
    b_delta = np.zeros((batches, ) + bias.shape)

    input_y = out_y = 0
    while (input_y + kernel_height) <= input_height:
        input_x = out_x = 0
        while (input_x + kernel_width) <= input_width:
            err = (error[:, out_y, out_x])[:, np.newaxis, np.newaxis]
            i = input[:, input_y:input_y + kernel_height, input_x:input_x + kernel_width]

            w_delta += (err * i)
            b_delta += err.reshape((batches, ) + bias.shape)

            bw_err = batch_weight * err
            #shallow copy
            bl_err = back_layer_error[:, input_y:input_y + kernel_height, input_x:input_x + kernel_width]
            bl_err += bw_err

            input_x += stride_x
            out_x += 1


        input_y += stride_y
        out_y += 1

    return back_layer_error, np.sum(w_delta, axis=0), np.sum(b_delta, axis=0)


def train(X, T, learning_rate, iterate):

    weight = np.random.normal(size=(3, 3))
    bias = np.zeros((1))

    input = padding(X)

    for i in range(iterate):

        y = convolution_forward(input, weight, bias)

        s = activation_forward(y)

        g = np.average((s - T)**2)

        if (i % 1000) == 0:
            print('epoch : ', i, '    mse : ', g)

        error = (s - T)

        error = activation_backward(s, error)

        error, weight_delta, bias_delta = convolution_backward(input, error, weight, bias)

        weight -= (learning_rate * weight_delta)
        bias -= (learning_rate * bias_delta)

    return weight, bias


train_x, train_t, test_x, test_t = loadDataSet()

weight, bias = train(train_x, train_t, learning_rate = 0.001, iterate = 30000)

input = padding(test_x)
y = convolution_forward(input, weight, bias)
s = activation_forward(y)

s = np.where(s >= 0.5, 1.0, s)
output = np.where(s < 0.5, 0.0, s)

equal = np.array_equal(output, test_t)
print('equal : ', equal)
