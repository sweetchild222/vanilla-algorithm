import numpy as np
from model.layer.weight_random import *
from model.layer.abs_layer import *
from model.gradient.creator import *
from model.activation.creator import *

class Dense(ABSLayer):

    def __init__(self, units, activation, weight_random, backward_layer, gradient):
        super(Dense, self).__init__(backward_layer)

        self.units = units
        self.activation = createActivation(activation)

        self.weight = self.createWeight(weight_random, (units, self.input_shape[0]))
        self.bias = np.zeros((units, 1))

        self.last_input = None

        self.gradient = createGradient(gradient)
        self.gradient.setShape(self.weight.shape, self.bias.shape)

    def createWeight(self, weight_random, size):

        fab_out = size[0]
        fab_in = size[1]

        return createWeightRandom(weight_random, fab_in, fab_out, size)

    def forward(self, input):

        self.last_input = input

        output = self.weight.dot(input) + self.bias

        return self.activation.forward(output)

    def backward(self, error, y):

        error = self.activation.backward(error, y)

        grain_weight = error.dot(self.last_input.T)
        grain_bias = np.sum(error, axis = 1).reshape(self.bias.shape)

        self.gradient.put(grain_weight, grain_bias)

        return self.weight.T.dot(error)

    def outputShape(self):
        return (self.units, )

    def updateGradient(self):

        deltaWeight = self.gradient.deltaWeight()
        detalBias = self.gradient.deltaBias()

        self.weight -= deltaWeight
        self.bias -= detalBias

        self.gradient.reset()