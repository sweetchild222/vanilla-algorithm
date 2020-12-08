def gradient_sgd():
    return {'type':'sgd', 'parameter':{'lr':0.01}}

def gradient_adam():
    return {'type':'adam', 'parameter':{'lr':0.001, 'beta1':0.95, 'beta2':0.95, 'exp':1e-7}}

def gradient_RMSprop():
    return {'type':'rmsProp', 'parameter':{'lr':0.001, 'beta':0.95, 'exp':1e-8}}



def activation_softmax():
    return {'type':'softmax', 'parameter':{}}

def activation_linear():
    return {'type':'linear', 'parameter':{}}

def activation_relu():
    return {'type':'relu', 'parameter':{}}

def activation_leakyRelu():
    return {'type':'leakyRelu', 'parameter':{'alpha':0.0001}}

def activation_sigmoid():
    return {'type':'sigmoid', 'parameter':{}}

def activation_elu():
    return {'type':'elu', 'parameter':{'alpha':0.0001}}

def activation_tanh():
    return {'type':'tanh', 'parameter':{}}


def template_complex(activation, gradient, input_shape, classes):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'convolution', 'parameter':{'filters':8, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':activation, 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'convolution', 'parameter':{'filters':8, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':activation, 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'flatten', 'parameter':{}},
        {'type':'dense', 'parameter':{'units':128, 'activation':activation, 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':classes, 'activation':activation_softmax(), 'gradient':gradient}}]

    return layers


def template_light(activation, gradient, input_shape, classes):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':activation, 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'flatten', 'parameter':{}},
        {'type':'dense', 'parameter':{'units':64, 'activation':activation, 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':classes, 'activation':activation_softmax(), 'gradient':gradient}}]

    return layers




'''
def template_light(activation, gradient, input_shape, classes):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'convolution', 'parameter':{'filters':8, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':activation, 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'convolution', 'parameter':{'filters':8, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':activation, 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'convolution', 'parameter':{'filters':1, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':activation, 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        #{'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'flatten', 'parameter':{}},
        {'type':'dense', 'parameter':{'units':64, 'activation':activation, 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':classes, 'activation':activation_softmax(), 'gradient':gradient}}]

    return layers

def template_light(activation, gradient, input_shape, classes):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'convolution', 'parameter':{'filters':1, 'kernel_size':(5, 5), 'strides':(1, 1), 'padding':True, 'activation':activation, 'gradient':gradient}},
        {'type':'convolution', 'parameter':{'filters':1, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':activation, 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'flatten', 'parameter':{}},
        {'type':'dense', 'parameter':{'units':64, 'activation':activation, 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':classes, 'activation':activation_softmax(), 'gradient':gradient}}]

    return layers

'''


def createModelTemplate(modelType, activationType, gradientType, input_shape, classes):

    modelTypeList = {'light':template_light, 'complex': template_complex}
    gradientTypeList = {'adam':gradient_adam, 'sgd':gradient_sgd, 'rmsProp':gradient_RMSprop}
    activationTypeList = {'elu':activation_elu, 'relu':activation_relu, 'leakyRelu':activation_leakyRelu, 'sigmoid':activation_sigmoid, 'tanh':activation_tanh, 'linear':activation_linear}

    template = modelTypeList[modelType]
    gradient = gradientTypeList[gradientType]
    activation = activationTypeList[activationType]

    return template(activation(), gradient(), input_shape, classes)
