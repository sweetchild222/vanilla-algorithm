from activation.abs_activation import *
from activation.leakyRelu import *
from activation.relu import *
from activation.elu import *
from activation.leakyRelu import *
from activation.sigmoid import *
from activation.softmax import *
from activation.tanh import *
from activation.linear import *


def createActivation(activation):

    type = activation['type']
    parameter = activation['parameter']

    typeClass = {'softmax':Softmax, 'relu':Relu, 'tanh':Tanh, 'leakyRelu':LeakyRelu, 'sigmoid':Sigmoid, 'elu':ELU, 'linear':Linear}

    return typeClass[type](**parameter)
