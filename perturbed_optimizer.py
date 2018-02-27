import numpy as np
import trigonometric as trig
import aux
import scipy.optimize as opt


def build_block_matrix(n):
    A_1 = np.ones((int(n/2), int(n/2)))
    A_2 = np.zeros((int(n/2), int(n/2)))
    A = np.vstack((np.hstack((A_1, A_2)), np.hstack((A_2, A_1))))
    return A


def minimize_gradient(A, init):
    """
    Returns the local minimizer found by the algorithm and the corresponding
    matrix X.

    Prints the value of local min and the corresponding local minimizer found
    by the gradient descent algorithm on the objective function of ||grad||_2.
    """

    dim, _ = A.shape
    # init = np.random.uniform(-2, 2, size=dim)
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

    return np.linalg.norm(trig.trig_grad(A, theta))


if __name__ == '__main__':
    n = 6
    sigma = 0.0001

    B = build_block_matrix(n)
    extra = sigma * np.ones((n, n))
    print("The block matrix B:\n{}".format(B))

    theta_1 = np.zeros(int(n/2))
    theta_2 = np.pi * np.ones(int(n/2))
    theta = np.hstack((theta_1, theta_2))
    # theta = np.zeros(n)
    print("The spurious solution on B:\n{}".format(theta))
    dim, _ = B.shape
    T = np.empty((dim, dim))
    for i in range(dim):
        T[i, :] = theta[i]
        T[i, :] = T[i, :] - theta
    cosT = np.cos(T)
    sinT = np.sin(T)

    grad = trig.trig_grad(B, theta)
    print("The gradient of function at spurious solution:\n{}".format(grad))
    hess = trig.trig_hess(B, theta)
    eigs = aux.sorted_eigenvalues(hess).ravel()
    print("The eigs of hessian of function at spurious solution:\n{}".format(eigs))

    A = B + extra
    print("The observation looks like block matrix:\n{}".format(A))

    grad_extra = trig.trig_grad(A, theta)
    print("The gradient of function at spurious solution:\n{}".format(grad_extra))
    hess_extra = trig.trig_hess(A, theta)
    eigs_extra = aux.sorted_eigenvalues(hess_extra).ravel()
    print("The eigs of hessian of function at spurious solution:\n{}".format(eigs_extra))

    theta_out, X = minimize_gradient(A, theta)
    print("Optimization result using min gradient:\n{}".format(theta_out))

    Q = trig.trig_bfgs(A, None, init=None)
    print("Optimization result using min function value:\n{}".format(Q.dot(Q.T)))
