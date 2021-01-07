from model.gradient.adam import *
from model.gradient.rms_prop import *
from model.gradient.sgd import *

def createGradient(gradient):

    type = gradient['type']
    parameter = gradient['parameter']
    typeClass = {'rmsProp':RMSprop, 'adam':Adam, 'sgd':SGD}

    return typeClass[type](**parameter)
