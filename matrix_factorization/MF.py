import numpy as np

def matrix_factorization(R, k, iteration):

    user_count, item_count = R.shape

    P = np.random.normal(size=(user_count, k))
    Q = np.random.normal(size=(item_count, k))

    bu = np.zeros(user_count)
    bi = np.zeros(item_count)

    for iter in range(iteration):
        for u in range(user_count):
            for i in range(item_count):
                r = R[u, i]
                if r >= 0:
                    error = prediction(P[u, :], Q[i, :], bu[u], bi[i]) - r

                    delta_Q, delta_bi = gradient(error, P[u, :])
                    delta_P, delta_bu = gradient(error, Q[i, :])

                    P[u, :] -= delta_P
                    bu[u] -= delta_bu

                    Q[i, :] -= delta_Q
                    bi[i] -= delta_bi

    return P.dot(Q.T) + bu[:, np.newaxis] + bi[np.newaxis:, ]

def gradient(error, weight):

    learning_rate = 0.005

    weight_delta = learning_rate * np.dot(weight.T, error)

    bias_delta = learning_rate * np.sum(error)

    return weight_delta, bias_delta


def prediction(P, Q, bu, bi):

	return P.dot(Q.T) + bu + bi


iteration = 100000
k = 3
R = np.array([
    [2, 8, 9, 1, 8],
    [8, 2, 1, 8, 1],
    [1, 5, -1, 1, 7],
    [7, 2, 1, 8, 1],
    [1, 8, 9, 2, 9],
    [9, 1, 2, -1, 2],
    [6, 1, 2, 7, 2]])

predicted_R = matrix_factorization(R, k, iteration)

print(predicted_R)
