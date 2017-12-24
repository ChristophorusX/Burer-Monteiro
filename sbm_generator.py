from math import log
import numpy as np
import aux


def _stochastic_block_model(n, p, q):
    z_sbm = np.hstack(
        (np.ones(int(n / 2)), -np.ones(int(n / 2)))).reshape(n, 1)
    R1_vec = aux.rounding_with_prob(
        np.random.random_sample(int(n / 2) * int(n / 2)), p)
    R3_vec = aux.rounding_with_prob(
        np.random.random_sample(int(n / 2) * int(n / 2)), p)
    R1_pre = R1_vec.reshape((int(n / 2), int(n / 2)))
    R3_pre = R3_vec.reshape((int(n / 2), int(n / 2)))
    for i in range(int(n / 2)):
        for j in range(int(n / 2)):
            if i > j:
                R1_pre[i, j] = R1_pre[j, i]
                R3_pre[i, j] = R3_pre[j, i]
            elif i == j:
                R1_pre[i, i] = 1
                R3_pre[i, i] = 1
    R2_vec = aux.rounding_with_prob(
        np.random.random_sample(int(n / 2) * int(n / 2)), q)
    R2_pre = R2_vec.reshape((int(n / 2), int(n / 2)))
    Y = np.vstack((np.hstack((R1_pre, R2_pre)), np.hstack((R2_pre.T, R3_pre))))
    return Y, z_sbm


def sbm_logarithm(n, a, b):
    p = a * log(n) / n
    q = b * log(n) / n
    Y, z_sbm = _stochastic_block_model(n, p, q)
    print('Adjacency matrix for SBM (logarithmic regime) is generated!')
    return Y, z_sbm


def sbm_linear(n, a, b):
    p = a / n
    q = b / n
    Y, z_sbm = _stochastic_block_model(n, p, q)
    print('Adjacency matrix for SBM (linear regime) is generated!')
    return Y, z_sbm

if __name__ == "__main__":
    print('Test starts for module sbm_generator...')
    Y_logarithm, z_logarithm = sbm_logarithm(10, 3, 9)
    Y_linear, z_linear = sbm_linear(10, 3, 9)
    print(Y_logarithm)
    print(z_logarithm)
    print(Y_linear)
    print(z_linear)
