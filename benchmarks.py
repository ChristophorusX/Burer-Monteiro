import numpy as np
import matplotlib.pyplot as plt
import monotone_adversaries as mono
import burer_monteiro as bm
import aux


def spectral_gap_increase(n, percentage, snr, noise_deviation):
    Y_adv, Y_sync, z = mono.monotone_sync(n, percentage, snr, noise_deviation)
    S_sync = dual_feasibility(Y_sync)
    S_adv = dual_feasibility(Y_adv)
    e_sync = aux.sorted_eigenvalues(S_sync)
    e_adv = aux.sorted_eigenvalues(S_adv)
    spectral_gap_increase = e_adv - e_sync
    return spectral_gap_increase


def dual_feasibility(Y):
    Q = bm.augmented_lagrangian(Y, 2)
    X = Q.dot(Q.T)
    S = np.diag(np.diag(Y.dot(X))) - Y
    return S

if __name__ == "__main__":
    increase_vec = spectral_gap_increase(1000, 0.5, 2, .00001)

    def compare(x):
        if np.real(x) >= 0:
            return 1
        else:
            return -1

    vec_compare = np.vectorize(compare)
    base_vec = np.arange(1000)
    plt.plot(base_vec, vec_compare(increase_vec).ravel())
    plt.show()
