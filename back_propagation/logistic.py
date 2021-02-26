import numpy as np

def activation_forward(x):

    return 1 / (1 + np.exp(-x))


def linear_forward(x, weight, bias):

    return np.dot(x, weight) + bias


def activation_backward(error, input):

    return error * input * (1.0 - input)


def linear_backward(error, weight):

    return np.dot(error, weight.T)


def train(X, T, learning_rate, iterate):

    inputNodes = X.shape[-1]
    outputNodes = T.shape[-1]

    #h1_weight = np.random.normal(size=(inputNodes, outputNodes))
    h1_weight = np.zeros((inputNodes, outputNodes))
    h1_bias = np.zeros((outputNodes))

    for i in range(iterate):

        y1 = linear_forward(X, h1_weight, h1_bias)
        s1 = activation_forward(y1)

        error = (s1 - T)

        error = activation_backward(error, s1)

        h1_weight_delta = np.dot(X.T, error)
        h1_bias_delta = np.sum(error, axis=0)

        error = linear_backward(error, h1_weight_delta)

        h1_weight -= (learning_rate * h1_weight_delta)
        h1_bias -= (learning_rate * h1_bias_delta)

    return h1_weight, h1_bias

X = np.array([[5]])
T = np.array([[1]])

h1_weight, h1_bias = train(X, T, learning_rate = 0.2, iterate = 4)

#print('weight  : ', h1_weight)
#print('bias  : ', h1_bias)

y1 = linear_forward(X, h1_weight, h1_bias)
s1 = activation_forward(y1)

print(s1)
