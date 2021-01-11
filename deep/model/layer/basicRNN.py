import numpy as np
from model.layer.weight_random import *
from model.layer.abs_layer import *
from model.gradient.creator import *
from model.activation.creator import *

class BasicRNN(ABSLayer):

    def __init__(self, units, activation, weight_random, backward_layer, gradient):
        super(BasicRNN, self).__init__(backward_layer)

        self.units = units
        self.activation = createActivation(activation)

        self.weight_x = self.createWeight(weight_random, (units, self.input_shape[0]))
        self.weight_h = self.createWeight(weight_random, (units, units))
        self.bias = np.zeros((units, 1))

        #self.h_list = [np.zeros((self.units, 1))]
        #self.input_list = []
        #self.h_size = 5
        #self.h_oldest = None
        #self.dhprev = np.zeros((self.units, 1))
        #self.error_list = []

        self.gradient_x = createGradient(gradient)
        self.gradient_x.setShape(self.weight_x.shape, self.bias.shape)

        self.gradient_h = createGradient(gradient)
        self.gradient_h.setShape(self.weight_h.shape, self.bias.shape)


    def createWeight(self, weight_random, size):

        fab_out = size[0]
        fab_in = size[1]

        return createWeightRandom(weight_random, fab_in, fab_out, size)

    def start(self):
        self.h_list = [np.zeros((self.units, 1))]
        self.input_list = []
        self.h_size = 8
        self.error_list = []

    def forward(self, input):

        self.input_list.append(input)

        h_prev = self.h_list[len(self.h_list) - 1]

        h_next = np.tanh(np.dot(self.weight_x, input) + np.dot(self.weight_h, h_prev) + self.bias)

        self.h_list.append(h_next)

        return h_next

    def backward(self, error, y):

        self.error_list.append(error)

        if len(self.h_list) >= self.h_size:
            dhprev = np.zeros((self.units, 1))

            #print('h : ', len(self.h_list))
            #print('e : ', len(self.error_list))

            for i in range(len(self.error_list) - 1, -1, -1):
                #print(len(self.error_list))
                #print(i)
                dh = self.error_list[i] + dhprev
                dhraw = (1 - self.h_list[i + 1]**2) * dh
                dhprev = np.dot(self.weight_h.T, dhraw)

            h_oldest = self.h_list.pop(0)
            input = self.input_list.pop(0)
            self.error_list.pop(0)

            grain_bias_x = dhraw
            grain_weight_x = np.dot(dhraw, input.T)
            grain_weight_h = np.dot(dhraw, h_oldest.T)

            self.gradient_x.put(grain_weight_x, grain_bias_x)
            self.gradient_h.put(grain_weight_h, grain_bias_x)

        #if len(self.error_list) > self.h_size:
        #self.dhnext = np.dot(self.weight_h.T, dhraw)

        return np.dot(self.weight_x.T, error)

    def outputShape(self):
        return (self.units, self.units)

    def updateGradient(self):

        deltaWeight_x = self.gradient_x.deltaWeight()
        detalBias_x = self.gradient_x.deltaBias()

        self.weight_x -= deltaWeight_x
        self.bias -= detalBias_x

        self.weight_h -= self.gradient_h.deltaWeight()

        self.gradient_x.reset()
        self.gradient_h.reset()
