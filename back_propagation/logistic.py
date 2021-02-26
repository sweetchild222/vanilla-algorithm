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
    weight = np.zeros((inputNodes, outputNodes))
    bias = np.zeros((outputNodes))

    for i in range(iterate):

        y = linear_forward(X, weight, bias)
        s = activation_forward(y)

        g = np.average((s - T)**2)

        if (i % 100) == 0:
            print(i, ' mse : ', g)

        error = (s - T)

        error = activation_backward(error, s)

        weight_delta = np.dot(X.T, error)
        bias_delta = np.sum(error, axis=0)

        error = linear_backward(error, weight)

        weight -= (learning_rate * weight_delta)
        bias -= (learning_rate * bias_delta)

    return weight, bias

X = np.array([[5]])
T = np.array([[1]])

weight, bias = train(X, T, learning_rate = 0.2, iterate = 500)

y = linear_forward(X, weight, bias)
s = activation_forward(y)

print('test : ', s)