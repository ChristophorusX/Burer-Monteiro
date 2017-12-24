import numpy as np
from math import log
import aux
import sync_generator as gen


def monotone_random(Y, delta):
    n, _ = Y.shape
    prob = delta * log(n) / n
    indicator = False
    diagnal = np.diag(Y)
    if diagnal.any() != 0:
        indicator = True
        V = Y - np.diag(diagnal)
    else:
        V = Y
    for i in range(0, int(n / 2)):
        for j in range(0, int(n / 2)):
            if V[i, j] == 0:
                if np.random.rand() < prob:
                    V[i, j] = 1
    for i in range(int(n / 2), n):
        for j in range(int(n / 2), n):
            if V[i, j] == 0:
                if np.random.rand() < prob:
                    V[i, j] = 1
    for i in range(0, int(n / 2)):
        for j in range(int(n / 2), n):
            if V[i, j] == 1:
                if np.random.rand() < prob:
                    V[i, j] = 0
    for i in range(int(n / 2), n):
        for j in range(0, int(n / 2)):
            if V[i, j] == 1:
                if np.random.rand() < prob:
                    V[i, j] = 0
    if indicator == True:
        V = V - np.diag(np.diag(V)) + np.diag(diagnal)
    else:
        V = V - np.diag(np.diag(V))
    print('Monotone adversary (random) has been performed...')
    return V


def monotone_hub(Y, portion):
    n, _ = Y.shape
    indicator = False
    diagnal = np.diag(Y)
    if diagnal.any() != 0:
        indicator = True
        V = Y - np.diag(diagnal)
    else:
        V = Y
    for i in range(1, int(n / 2)):
        if V[i, 0] == 0:
            if np.random.rand() < portion:
                V[i, 0] = 1
    for i in range(int(n / 2), n):
        if V[i, 0] == 1:
            if np.random.rand() < portion:
                V[i, 0] = 0
    if indicator == True:
        V = V - np.diag(np.diag(V)) + np.diag(diagnal)
    else:
        V = V - np.diag(np.diag(V))
    print('Monotone adversary (hub) has been performed...')
    return V


def monotone_sync(n, percentage, snr, noise_deviation, generator=gen.synchronization_normalized):
    Y, z = generator(n, percentage, snr)
    Y_sync = Y.copy()
    Z = z.dot(z.T)
    for i in range(n):
        for j in range(n):
            if Z[i, j] > 0 and i < j:
                noise = abs(np.random.normal(0, noise_deviation))
                Y[i, j] += noise
            elif i < j:
                noise = abs(np.random.normal(0, noise_deviation))
                Y[i, j] -= noise
    for i in range(n):
        for j in range(n):
            if i > j:
                Y[i, j] = Y[j, i]
    print('Monotone adversary (Gaussian) has been performed...')
    return Y, Y_sync, z


if __name__ == "__main__":
    test_vec = aux.rounding_with_prob(np.random.random_sample(100), .5)
    test_matrix = test_vec.reshape(10, 10)
    print(test_matrix)
    # result = monotone_random(test_matrix, .5)
    # print(result)
    print(monotone_sync(10, .5, 10, 1)[0])
