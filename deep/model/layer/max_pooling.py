import numpy as np
from model.layer.abs_layer import *

class MaxPooling(ABSLayer):

    def __init__(self, pool_size, strides, backward_layer):
        super(MaxPooling, self).__init__(backward_layer)

        self.pool_size = pool_size
        self.strides = pool_size if strides == None else strides
        self.last_input = None


    def forward(self, input):

        self.last_input = input

        (input_colors, input_height, input_width) = input.shape
        (pool_height, pool_width) = self.pool_size
        (stride_y, stride_x) = self.strides

        height = int(input_height // stride_y)
        width = int(input_width // stride_x)

        output = np.zeros((input_colors, height, width))

        for color in range(input_colors):
            input_y = out_y = 0
            while (input_y + pool_height) <= input_height:
                input_x = out_x = 0
                while (input_x + pool_width) <= input_width:
                    output[color, out_y, out_x] = np.max(input[color, input_y:input_y + pool_height, input_x:input_x + pool_width])
                    input_x += stride_x
                    out_x += 1
                input_y += stride_y
                out_y += 1

        return output


    def nanargmax(self, array):
        idx = np.nanargmax(array)
        return np.unravel_index(idx, array.shape)


    def backward(self, error, y):

        (input_colors, input_height, input_width) = self.input_shape
        (pool_height, pool_width) = self.pool_size
        (stride_y, stride_x) = self.strides

        output = np.zeros(self.input_shape)

        for color in range(input_colors):
            input_y = out_y = 0
            while (input_y + pool_height) <= input_height:
                input_x = out_x = 0
                while (input_x + pool_width) <= input_width:
                    (a, b) = self.nanargmax(self.last_input[color, input_y:input_y + pool_height, input_x:input_x + pool_width])
                    output[color, input_y + a, input_x + b] = error[color, out_y, out_x]
                    input_x += stride_x
                    out_x += 1
                input_y += stride_y
                out_y += 1

        return output


    def beginBatch(self):
        pass


    def endBatch(self):
        pass


    def outputShape(self):

        calc_shape = ((self.input_shape[1] // self.strides[0]), (self.input_shape[2] // self.strides[1]))

        return (self.input_shape[0],) + calc_shape
