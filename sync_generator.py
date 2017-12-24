from math import sqrt
import numpy as np
import aux


def _synchronization(n, percentage, snr):
    z = aux.rounding_with_prob(np.random.random_sample(n), percentage)
    z = 2 * z.reshape(n, 1) - 1
    ground_truth = z.dot(z.T)
    gaussian = np.random.normal(0, sqrt(2) / 2, n**2).reshape(n, n)
    sym_gaussian = gaussian + gaussian.T
    W = sym_gaussian - np.diag(np.diag(sym_gaussian))
    sigma = sqrt(n) / snr
    Y = ground_truth + sigma * W
    return Y, z


def synchronization_normalized(n, percentage, snr):
    Y, z = _synchronization(n, percentage, snr)
    Y_normalized = snr / n * Y
    print('Observed Z2 synchronization matrix (normalized) is generated!')
    return Y_normalized, z


def synchronization_usual(n, percentage, snr):
    print('Observed Z2 synchronization matrix (usual) is generated!')
    return _synchronization(n, percentage, snr)


if __name__ == "__main__":
    print('Test starts for module sync_generator...')
    Y_usual, z_usual = synchronization_usual(10, 0.5, 15000)
    Y_normalized, z_normalized = synchronization_normalized(10, 0.5, 15000)
    print(Y_usual)
    print(z_usual)
    print(Y_normalized)
    print(z_normalized)
    print(aux.laplacian_eigs(Y_usual, z_usual))
    print(np.linalg.norm(Y_usual - z_usual.dot(z_usual.T)))
