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
    #a = np.array([[[1,1,1], [1,1,1], [1,1,1]], [[2,2,2], [2,2,2], [2,2,2]], [[3,3,3], [3,3,3], [3,3,3]], [[4,4,4], [4,4,4], [4,4,4]]])
    #b = np.array([[2,2,2], [2,2,2], [2,2,2]])

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


def unravel_indices(input):

    indices = np.nanargmax(input)

    indices = np.unravel_index(indices, input.shape)

    return indices


def pooling_backward(input, error, pool_size, pool_stride):

    (batches, input_height, input_width) = input.shape
    (pool_height, pool_width) = pool_size
    (stride_y, stride_x) = pool_stride

    back_layer_error = np.zeros(input.shape)

    input_y = out_y = 0
    while (input_y + pool_height) <= input_height:
        input_x = out_x = 0
        while (input_x + pool_width) <= input_width:

            for b in range(batches):
                (unravel_y, unravel_x) = unravel_indices(input[b, input_y:input_y + pool_height, input_x:input_x + pool_width])
                back_layer_error[b, input_y + unravel_y, input_x + unravel_x] = error[b, out_y, out_x]

            input_x += stride_x
            out_x += 1

        input_y += stride_y
        out_y += 1

    return back_layer_error



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


def train(X, T, convol_size, convol_stride, pool_size, pool_stride, learning_rate, iterate):

    weight = np.random.normal(size=convol_size)
    bias = np.zeros((1))

    for i in range(iterate):

        y = convolution_forward(X, weight, bias, convol_stride)

        s = activation_forward(y)

        p = pooling_forward(s, pool_size, pool_stride)

        g = np.average((p - T)**2)

        if ((i + 1) % 1000) == 0:
            print('epoch : ', i + 1, '    mse : ', g)
            if (iterate - i) == 1:

                np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
                #all_x = np.where(s >= 0.5, 1.0, s)
                #all_x = np.where(all_x < 0.5, 0.0, all_x)
                print('s', s)
                #print('all x', all_x)
                print('p', p)

        error = (p - T)

        error = pooling_backward(s, error, pool_size, pool_stride)

        error = activation_backward(s, error)

        error, weight_delta, bias_delta = convolution_backward(X, error, weight, bias, convol_stride)

        weight -= (learning_rate * weight_delta)
        bias -= (learning_rate * bias_delta)

    return weight, bias


convol_size = (3, 3)
convol_stride = (1, 1)
pool_size = (2, 2)
pool_stride = pool_size

train_x, train_t, test_x, test_t = loadDataSet()

train_t = pooling_forward(train_t, pool_size, pool_stride)
test_t = pooling_forward(test_t, pool_size, pool_stride)


weight, bias = train(train_x, train_t, convol_size, convol_stride, pool_size, pool_stride, learning_rate = 0.001, iterate = 100000)

y = convolution_forward(test_x, weight, bias, convol_stride)
s = activation_forward(y)
p = pooling_forward(s, pool_size, pool_stride)

p = np.where(p >= 0.5, 1.0, p)
output = np.where(p < 0.5, 0.0, p)

print('out', output)
print('test', test_t)

equal = np.array_equal(output, test_t)
print('equal : ', equal)
