import numpy as np
from gradient.abs_gradient import *

class Adam(ABSGradient):

    def __init__(self, lr, beta1, beta2, exp):
        super(Adam, self).__init__(lr)

        self.beta1 = beta1
        self.beta2 = beta2

        self.delta_weight = None
        self.delta_bias = None

        self.vector_weight = None
        self.vector_bias = None

        self.exp = exp

        self.size = 0

    def setShape(self, weightShape, biasShape):

        self.delta_weight = np.zeros(weightShape)
        self.delta_bias = np.zeros(biasShape)

        self.vector_weight = np.zeros(weightShape)
        self.vector_bias = np.zeros(biasShape)

        self.size = 0

    def put(self, grain_weight, grain_bias):

        self.delta_weight += grain_weight
        self.delta_bias += grain_bias

        self.size += 1

    def deltaWeight(self):

        avg_delta = self.delta_weight / self.size

        self.vector_weight = self.beta1 * self.vector_weight + (1 - self.beta2) * (avg_delta)**2

        return self.lr * (avg_delta)/(np.sqrt(self.vector_weight) + self.exp)

    def deltaBias(self):

        avg_delta = self.delta_bias / self.size

        self.vector_bias = self.beta1 * self.vector_bias + (1 - self.beta2) * (avg_delta)**2

        return self.lr * (avg_delta)/(np.sqrt(self.vector_bias) + self.exp)

    def reset(self):

        self.delta_weight = np.zeros(self.delta_weight.shape)
        self.delta_bias = np.zeros(self.delta_bias.shape)
        self.size = 0
