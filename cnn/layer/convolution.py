import numpy as np
from layer.abs_layer import *
from gradient.creator import *
from activation.creator import *

class Convolution(ABSLayer):

    def __init__(self, filters, kernel_size, strides, padding, activation, backward_layer, gradient):
        super(Convolution, self).__init__(backward_layer)

        self.strides = strides
        self.padding_size = self.paddingSize(kernel_size[0], kernel_size[1]) if padding else (0,0)

        self.weight = self.initWeight((filters, self.input_shape[0], kernel_size[0], kernel_size[1]))
        self.bias = np.zeros((filters, 1))

        self.gradient = createGradient(gradient)
        self.gradient.setShape(self.weight.shape, self.bias.shape)

        self.last_output = None
        self.last_input = None
        self.activation = createActivation(activation)


    def initWeight(self, size, scale = 1.0):

        stddev = scale/np.sqrt(np.prod(size))
        return np.random.normal(loc = 0, scale = stddev, size = size)

    def appendPadding(self, input):

        size = self.padding_size

        rows = []
        for i in input:
            rows.append(np.pad(i, ((size[0], size[0]),(size[1], size[1])), 'constant', constant_values=0))

        return np.array(rows)

    def forward(self, input):

        input = self.appendPadding(input)

        self.last_input = input

        (filters, colors, kernel_height, kernel_width) = self.weight.shape
        (input_colors, input_height, input_width) = self.input_shape
        (stride_y, stride_x) = self.strides

        assert colors == input_colors, "filter miss match"

        output = np.zeros(self.outputShape())

        for filter in range(filters):
            input_y = out_y = 0
            while (input_y + kernel_height) <= input_height:
                input_x = out_x = 0
                while (input_x + kernel_width) <= input_width:
                    output[filter, out_y, out_x] = np.sum(self.weight[filter] * input[:, input_y:input_y + kernel_height, input_x:input_x + kernel_width]) + self.bias[filter]
                    input_x += stride_x
                    out_x += 1

                input_y += stride_y
                out_y += 1

        return self.activation.forward(output)

    def backward(self, error, y):

        error = self.activation.backward(error, y)

        (filters, colors, kernel_height, kernel_width) = self.weight.shape
        (input_colors, input_height, input_width) = self.input_shape
        (stride_y, stride_x) = self.strides

        output = np.zeros(self.input_shape)
        grain_weight = np.zeros(self.weight.shape)
        grain_bias = np.zeros(self.bias.shape)

        for filter in range(filters):
            input_y = out_y = 0
            while (input_y + kernel_height) <= input_height:
                input_x = out_x = 0
                while (input_x + kernel_width) <= input_width:
                    grain_weight[filter] += error[filter, out_y, out_x] * self.last_input[:, input_y:input_y + kernel_height, input_x:input_x + kernel_width]
                    output[:, input_y:input_y + kernel_height, input_x:input_x + kernel_width] += error[filter, out_y, out_x] * self.weight[filter]
                    input_x += stride_x
                    out_x += 1
                input_y += stride_y
                out_y += 1

            grain_bias[filter] = np.sum(error[filter])

        self.gradient.put(grain_weight, grain_bias)

        return output

    def paddingSize(self, kernel_height, kernel_width):

        return ((kernel_height - 1) // 2, (kernel_width - 1) // 2)

    def outputShape(self):

        (filters, colors, kernel_height, kernel_width) = self.weight.shape
        (stride_y, stride_x) = self.strides

        numerator_height = ((self.padding_size[0] * 2) - kernel_height) + self.input_shape[1]
        numerator_width = ((self.padding_size[1] * 2) - kernel_width) + self.input_shape[2]

        calc_shape = ((numerator_height // stride_y) + 1, (numerator_width // stride_x) + 1)

        return (filters,) + calc_shape

    def updateGradient(self):

        deltaWeight = self.gradient.deltaWeight()
        deltaBias = self.gradient.deltaBias()

        self.weight -= deltaWeight
        self.bias -= deltaBias

        self.gradient.reset()
