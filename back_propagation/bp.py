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

    weight = np.random.normal(size=(inputNodes, outputNodes))
    bias = np.zeros((outputNodes))

    for i in range(iterate):

        y1 = linear_forward(X, weight, bias)
        s1 = activation_forward(y1)

        error = (s1 - T)

        if (i % 1000) is 0:
            print(i, ' : ', s1)
            #print('T : ', T)
            #print('S1 : ', s1)
            #print('e : '  , error)
            #print(i, ' :', loss)

        error = activation_backward(error, s1)

        weight_delta = np.dot(X.T, error)
        bias_delta = np.sum(error, axis=0)

        error = linear_backward(error, weight)

        weight -= (learning_rate * weight_delta)
        bias -= (learning_rate * bias_delta)

    return weight, bias

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([[0], [1], [1], [0]])

weight, bias = train(X, T, learning_rate = 0.01, iterate = 10000)

print('weight  : ', weight)
print('bias  : ', bias)
