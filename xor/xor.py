import numpy as np

def activation(y):
    y = 1 / (1 + np.exp(-y))
    return y

def h1(x):
    weight = np.array([[20, 20], [-20, -20]])
    bias = np.array([-10, 30])
    y = np.dot(x, weight.T) + bias
    return activation(y)

def h2(x):
    weight = np.array([20, 20])
    bias = np.array([-30])
    y = np.dot(x, weight.T) + bias
    return activation(y)

x = np.array([[0,0], [1, 1], [0, 1], [1, 0]])

y = h2(h1(x))

for i in range(len(x)):
    print(x[i], ' => ', y[i])
