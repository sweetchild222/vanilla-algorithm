import numpy as np
from model.gradient.abs_gradient import *

class SGD(ABSGradient):

    def __init__(self, lr):
        super(SGD, self).__init__(lr)

        self.delta_weight = None
        self.delta_bias = None

        self.size = 0

    def setShape(self, weightShape, biasShape):

        self.delta_weight = np.zeros(weightShape)
        self.delta_bias = np.zeros(biasShape)

        self.size = 0

    def put(self, grain_weight, grain_bias):

        self.delta_weight += grain_weight
        self.delta_bias += grain_bias

        self.size += 1

    def deltaWeight(self):

        avg_delta = self.delta_weight / self.size

        return self.lr * avg_delta

    def deltaBias(self):

        avg_delta = self.delta_bias / self.size

        return self.lr * avg_delta

    def reset(self):

        self.delta_weight = np.zeros(self.delta_weight.shape)
        self.delta_bias = np.zeros(self.delta_bias.shape)
        self.size = 0
