import numpy as np


def rounding_with_prob(vec, p):
    """
    Rounds a vector to 0 and 1 according to the value of its entries,
    with threshold p. If vec is a vector uniformly generated from 0 and 1
    then p indicates the probility of being 1.
    """

    for index in range(vec.size):
        vec[index] = vec[index] < p
    return vec


def demean(A, p, q):
    """
	Demeans a SBM observation by generating probability.
	"""

    n, _ = A.shape
    one_vector = create_one_vector(n)
    A_bm = A - (p + q) / 2 * (one_vector.dot(one_vector.T))
    print('Demeaned the SBM matrix into non-biased matrix for BM...')
    return A_bm


def demean_adversary(V):
    """
	Demeans an observation for every row according to its mean.
	"""

    n, _ = V.shape
    col_sum = np.sum(V, axis=1).reshape(-1, 1)
    one_vector = create_one_vector(n)
    V_bm = V - col_sum.dot(one_vector.reshape(1, -1)) / n
    print('Demeaned the adversary SBM matrix into non-biased matrix for BM...')
    return V_bm


def create_one_vector(n):
    """
	Generates a column vector of 1.
	"""

    return np.ones(n).reshape((-1, 1))


def laplacian_eigs(Y, z):
    """
    Returns the eigenvalues of the Laplacian like form of Y as a column
    vector from small to large.
    """

    compo = (np.diag(z.ravel()).dot(Y)).dot(np.diag(z.ravel()))
    D = np.diag(np.sum(compo, axis=1))
    L = D - Y
    w, _ = np.linalg.eig(L)
    eigs = np.sort(w, axis=None)
    print('Successfully computed the Laplacian eigenvalues...')
    n, _ = Y.shape
    return eigs.reshape(n, 1)


def laplacian(Y):
    """
	Returns a Laplacian like matrix by using the trivial summation of rows.
	"""

    D = np.diag(np.sum(Y, axis=1))
    L = D - Y
    return L


def normalize(vec):
    """
	Normalizes a vector.
	"""

    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def sorted_eigenvalues(S):
    """
	Returns sorted eigenvalues of matrix S from small to large as a column vector.
	"""

    return np.sort(np.linalg.eigvals(S)).reshape((-1, 1))


def error_rate(labels, true_labels):
    """
	Returns the error rate of the guess compared to true labels up to global flip.
	"""

    diff1 = labels - true_labels
    diff2 = labels + true_labels
    rate1 = np.linalg.norm(diff1, 1) / 2 / labels.shape[0]
    rate2 = np.linalg.norm(diff2, 1) / 2 / labels.shape[0]
    return min(rate1, rate2)


def hess_equiv(A, Q):
    """
	Returns the equivalent Hessian when dealing with rank 2 problems.
	"""

    first = np.diag(np.diag((A.dot(Q)).dot(Q.T)))
    second = A * (Q.dot(Q.T))
    return first - second


def frobenius_distance(A, B):
    """
	Returns the Frobenius norm between two matrices.
	"""
    return np.linalg.norm(A - B)



if __name__ == "__main__":
    Y = np.diag([1, 1, 3, 4]) + [[2, 5, 3, 5],
                                 [4, 5, 1, 2], [4, 5, 1, 2], [4, 5, 1, 2]]
    print(Y)
    S = Y + Y.T
    print(S)
    # eigs = laplacian_eigs(np.diag([1,1,3,4])+ [[2,5,3,5], [4,5,1,2], [4,5,1,2], [4,5,1,2]])
    # L = laplacian(Y)
    # print(eigs)
    # print(L)
    norm = normalize(Y[1, :])
    print(norm)
    print(create_one_vector(10))
    print(sorted_eigenvalues(S))
