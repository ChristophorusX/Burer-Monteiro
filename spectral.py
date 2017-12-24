import numpy as np
from scipy import cluster
import aux
import sbm_generator


def spectral_clustering_unnormalized(S, k):
    n, _ = S.shape
    L = aux.laplacian(S)
    w, V = np.linalg.eigh(L)
    index = np.argsort(w)
    U = _stack_vec(V, k, index)
    _, label = cluster.vq.kmeans2(U, k)
    return label.reshape((-1, 1))


def spectral_clustering_rw(S, k):
    L = aux.laplacian(S)
    D = np.diag(np.sum(S, axis=1))
    w, V = np.linalg.eigh(D.dot(L))
    index = np.argsort(w)
    U = _stack_vec(V, k, index)
    _, label = cluster.vq.kmeans2(U, k)
    return label.reshape((-1, 1))


def spectral_clustering_sym(S, k):
    n, _ = S.shape
    L = aux.laplacian(S)
    D = np.diag(np.sum(S, axis=1))
    D_half = np.sqrt(D)
    D_minus_half = np.diag(1 / np.diag(D_half))
    w, V = np.linalg.eigh(D_minus_half.dot(L).dot(D_minus_half))
    index = np.argsort(w)
    U = _stack_vec(V, k, index)
    for row in U:
        row = aux.normalize(row)
    print(U)
    _, label = cluster.vq.kmeans2(U, k)
    return label.reshape((-1, 1))


def _stack_vec(V, k, index):
    U = V[:, index[0]].reshape(-1, 1)
    for i in range(1, k):
        U = np.hstack((U, V[:, index[i]].reshape(-1, 1)))
    return U


if __name__ == "__main__":
    S, z = sbm_generator.sbm_linear(10, 9, 2)
    print(S)
    print(z)
    # spectral_clustering_unnormalized(S, 2)
    # spectral_clustering_rw(S, 2)
    spectral_clustering_sym(S, 2)
