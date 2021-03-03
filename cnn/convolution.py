import numpy as np
import matplotlib.pyplot as plt
from loader import *


def activation_forward(x):

    return 1 / (1 + np.exp(-x))


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

            input_x += stride_x
            out_x += 1

        input_y += stride_y
        out_y += 1



    return w_delta, np.sum(w_delta, axis=0), np.sum(b_delta, axis=0)


def train(X, T, learning_rate, iterate):

    weight = np.random.normal(size=(3, 3))
    bias = np.zeros((1))

    pad_size = 1

    input = np.pad(X, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=0)

    for i in range(iterate):

        y = convolution_forward(input, weight, bias)
        s = activation_forward(y)

        #print(s)
        #print(T)

        g = np.average((s - T)**2)

        if (i % 500) == 0:
            print(i, ' mse : ', g)
            #print('s', s[0])
            #print('t', T[0])

        error = (s - T)

        error = activation_backward(s, error)

        #weight_delta = np.dot(X.T, error)
        #bias_delta = np.sum(error, axis=0)

        error, weight_delta, bias_delta = convolution_backward(input, error, weight, bias)

        #print(weight.shape)
        #print(weight_delta.shape)

        weight -= (learning_rate * weight_delta)
        bias -= (learning_rate * bias_delta)

    return weight, bias

#X = np.array([[0, 0], [1, 1]])
#T = np.array([[0], [1]])

#weight, bias = train(X, T, learning_rate = 0.01, iterate = 50000)


train_x, train_t, test_x, test_t = loadDataSet()

weight, bias = train(train_x, train_t, learning_rate = 0.001, iterate = 50000)



#y1 = linear_forward(X, h1_weight, h1_bias)
#s1 = activation_forward(y1)

#y2 = linear_forward(s1, h2_weight, h2_bias)
#s2 = activation_forward(y2)

#print('test : ', s2)
