import numpy as np
import basin_of_attraction as basin
import trigonometric as trig
import scipy.optimize as opt
import aux


def minimize_gradient(A, z):
    """
    Returns the local minimizer found by the algorithm and the corresponding
    matrix X.

    Prints the value of local min and the corresponding local minimizer found
    by the gradient descent algorithm on the objective function of ||grad||_2.
    """

    dim, _ = A.shape
    init = np.random.uniform(-2, 2, size=dim)
    result = opt.minimize(
        fun=lambda theta: gradient_norm_func(A, theta), x0=init)
    theta = result.x
    opt_val = result.fun
    T = np.empty((dim, dim))
    for i in range(dim):
        T[i, :] = theta[i]
        T[i, :] = T[i, :] - theta
    X = np.cos(T)
    print("The local minimum is: {}".format(opt_val))
    print("The corresponding minimizer is: \n{}".format(X))

    return theta, X


def gradient_norm_func(A, theta):
    """
    Returns the 2 norm of the gradient vector (more precisely,
    1/2 of the gradient vector).
    """

    return np.linalg.norm(trig.trig_grad(A, theta))


def evaluate_curvature(A, X):
    """
    Returns the Hessian given the observation A and a point X on the manifold.
    """

    dim, _ = A.shape
    one = np.ones(dim).reshape((-1, 1))
    hess = A * X - np.diag((A * X).dot(one))
    return hess


def smallest_nonzero_curv(hess):
    """Returns the smallest nonzero curvature for the given Hessian."""

    curv_arr = aux.sorted_eigenvalues(hess).ravel()
    curv = curv_arr[0]
    if curv > - 10 ** (-6):
        curv = curv_arr[1]
    return curv


if __name__ == '__main__':
    A, z = basin.get_observation(10, 1, 'positive-rows')
    theta, X = minimize_gradient(A, z)
    hessian = evaluate_curvature(A, X)
    print("The smallest curvature is : {}".format(smallest_nonzero_curv(hessian)))
