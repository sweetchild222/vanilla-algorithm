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

    h1_units = 5
    h1_weight = np.random.normal(size=(inputNodes, h1_units))
    h1_bias = np.zeros((h1_units))

    h2_weight = np.random.normal(size=(h1_units, outputNodes))
    h2_bias = np.zeros((outputNodes))

    for i in range(iterate):

        y1 = linear_forward(X, h1_weight, h1_bias)
        s1 = activation_forward(y1)

        y2 = linear_forward(s1, h2_weight, h2_bias)
        s2 = activation_forward(y2)

        error = (s2 - T)

        if (i % 10000) is 0:
            print(i, ' : ', s2)
            #print('T : ', T)
            #print('S1 : ', s1)
            #print('e : '  , error)
            #print(i, ' :', loss)

        error = activation_backward(error, s2)

        h2_weight_delta = np.dot(s1.T, error)
        h2_bias_delta = np.sum(error, axis=0)

        error = linear_backward(error, h2_weight_delta)

        h2_weight -= (learning_rate * h2_weight_delta)
        h2_bias -= (learning_rate * h2_bias_delta)

        error = activation_backward(error, s1)

        h1_weight_delta = np.dot(X.T, error)
        h1_bias_delta = np.sum(error, axis=0)

        error = linear_backward(error, h1_weight_delta)

        h1_weight -= (learning_rate * h1_weight_delta)
        h1_bias -= (learning_rate * h1_bias_delta)

    return h1_weight, h1_bias, h2_weight, h2_bias

X = np.array([[0, 0], [1, 1]])
T = np.array([[0], [1]])

h1_weight, h1_bias, h2_weight, h2_bias = train(X, T, learning_rate = 0.01, iterate = 10000000)

print('h1 weight  : ', h1_weight)
print('h1 bias  : ', h1_bias)

print('h2 weight  : ', h2_weight)
print('h2 bias  : ', h2_bias)
