import numpy as np
import burer_monteiro as bm
import basin_of_attraction as basin
import aux
from scipy import optimize as opt


def trig_objective_function(A, theta):
    """
    Returns the function value in trigonometric parameterization.
    """

    dim, _ = A.shape
    T = np.empty((dim, dim))
    for i in range(dim):
        T[i, :] = theta[i]
        T[i, :] = T[i, :] - theta
    return np.trace(A.dot(np.cos(T)))


def trig_grad(A, theta):
    """
    Returns the function gradient in trigonometric parameterization.
    """

    dim, _ = A.shape
    T = np.empty((dim, dim))
    for i in range(dim):
        T[i, :] = theta[i]
        T[i, :] = T[i, :] - theta
    one = np.ones(dim).reshape((-1, 1))
    hadamard = (A * np.sin(T)).transpose()
    return hadamard.dot(one).ravel()


def trig_hess(A, theta):
    """
    Returns the function Hessian in trigonometric parameterization.
    """

    dim, _ = A.shape
    T = np.empty((dim, dim))
    for i in range(dim):
        T[i, :] = theta[i]
        T[i, :] = T[i, :] - theta
    one = np.ones(dim).reshape((-1, 1))
    return A * np.cos(T) - np.diag((A * np.cos(T)).dot(one))


def trig_hessp(A, theta, p):
    """
    Returns the function Hessian in trigonometric parameterization
    as an inner product function.
    """

    H = trig_hess(A, theta)
    return p.dot(H)


def recover_solution(theta):
    """
    Recovers triangular matrix Q from trigonometric parameterization.
    """

    Q = np.hstack((np.cos(theta).reshape((-1, 1)),
                   np.sin(theta).reshape((-1, 1))))
    return Q


def trig_trust_region(A, z):
    """
    Returns the optimization result Q under trust region with
    trigonometric parameterization.
    """

    dim, _ = A.shape
    init = np.random.uniform(-10, 10, size=dim)
    optimizer = bm.minimize_with_trust(fun=lambda theta: -trig_objective_function(A, theta),
                                       x0=init, n_rows=1, plotting=None, printing=None,
                                       jac=lambda theta: -trig_grad(A, theta),
                                       hessp=lambda theta, p: -
                                       trig_hessp(A, theta, p),
                                       hess=lambda theta: -trig_hess(A, theta))
    theta = optimizer.x
    Q = recover_solution(theta)
    return Q


def trig_bfgs(A, z, init=None):
    """
    Returns the optimization result Q under BFGS with
    trigonometric parameterization.
    """

    dim, _ = A.shape
    if init is None:
        init = np.random.uniform(-10, 10, size=dim)
    optimizer = opt.minimize(fun=lambda theta: -trig_objective_function(A, theta),
                             x0=init, jac=lambda theta: -trig_grad(A, theta),
                             method='BFGS')
    theta = optimizer.x
    Q = recover_solution(theta)
    return Q


if __name__ == '__main__':
    A, z = basin.get_observation(10, 3, 'sync')
    z = z.reshape((-1, 1))
    # print(trig_objective_function(A, np.ones(10)))
    # print(trig_grad(A, np.ones(10)))
    # print(trig_hess(A, np.ones(10)))
    Q = trig_trust_region(A, z)
    # Q = trig_bfgs(A, z)
    diff_norm = aux.frobenius_distance(z.dot(z.T), Q.dot(Q.T))
    print("The Frobenius distance to the ground truth is: {}".format(diff_norm))
