from gradient.adam import *
from gradient.rms_prop import *
from gradient.sgd import *

def createGradient(gradient):

    type = gradient['type']
    parameter = gradient['parameter']
    typeClass = {'rmsProp':RMSprop, 'adam':Adam, 'sgd':SGD}

    return typeClass[type](**parameter)
