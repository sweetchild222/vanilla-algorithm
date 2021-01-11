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




def weight_random_glorot_normal():
    return {'type':'glorot', 'random':'normal'}

def weight_random_glorot_uniform():
    return {'type':'glorot', 'random':'uniform'}

def weight_random_he_normal():
    return {'type':'he', 'random':'normal'}

def weight_random_he_uniform():
    return {'type':'he', 'random':'uniform'}

def weight_random_lecun_normal():
    return {'type':'lecun', 'random':'normal'}

def weight_random_lecun_uniform():
    return {'type':'lecun', 'random':'uniform'}


def template_complex(activation, weight_random, gradient, input_shape, classes):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'flatten', 'parameter':{}},
        {'type':'dense', 'parameter':{'units':1024, 'activation':activation, 'weight_random':weight_random, 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':512, 'activation':activation, 'weight_random':weight_random, 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':256, 'activation':activation, 'weight_random':weight_random, 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':classes, 'activation':activation_softmax(), 'weight_random':weight_random, 'gradient':gradient}}]

    return layers


def template_light(activation, weight_random, gradient, input_shape, classes):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'flatten', 'parameter':{}},
        {'type':'basicRNN', 'parameter':{'units':512, 'activation':activation, 'weight_random':weight_random, 'gradient':gradient}},
        #{'type':'dense', 'parameter':{'units':64, 'activation':activation, 'weight_random':weight_random, 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':classes, 'activation':activation_softmax(), 'weight_random':weight_random, 'gradient':gradient}}]

    return layers



def createModelTemplate(modelType, activationType, weightType, weightRandomType, gradientType, input_shape, classes):

    modelTypeList = {'light':template_light, 'complex': template_complex}
    activationTypeList = {'elu':activation_elu, 'relu':activation_relu, 'leakyRelu':activation_leakyRelu, 'sigmoid':activation_sigmoid, 'tanh':activation_tanh, 'linear':activation_linear}
    gradientTypeList = {'adam':gradient_adam, 'sgd':gradient_sgd, 'rmsProp':gradient_RMSprop}
    weightRandomTypeList = {'glorot_normal':weight_random_glorot_normal, 'glorot_uniform':weight_random_glorot_uniform, 'he_normal':weight_random_he_uniform, 'he_uniform':weight_random_he_uniform, 'lecun_normal':weight_random_lecun_normal, 'lecun_uniform':weight_random_lecun_uniform}

    template = modelTypeList[modelType]
    gradient = gradientTypeList[gradientType]
    activation = activationTypeList[activationType]
    weightRandom = weightRandomTypeList[weightType + '_' + weightRandomType]

    return template(activation(), weightRandom(), gradient(), input_shape, classes)
