from model.activation.abs_activation import *
from model.activation.leakyRelu import *
from model.activation.relu import *
from model.activation.elu import *
from model.activation.leakyRelu import *
from model.activation.sigmoid import *
from model.activation.softmax import *
from model.activation.tanh import *
from model.activation.linear import *


def createActivation(activation):

    type = activation['type']
    parameter = activation['parameter']

    typeClass = {'softmax':Softmax, 'relu':Relu, 'tanh':Tanh, 'leakyRelu':LeakyRelu, 'sigmoid':Sigmoid, 'elu':ELU, 'linear':Linear}

    return typeClass[type](**parameter)
