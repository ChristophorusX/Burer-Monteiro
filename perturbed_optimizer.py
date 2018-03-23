import numpy as np
import trigonometric as trig
import aux
import scipy.optimize as opt
import scipy as sp
import burer_monteiro as bm


def build_block_matrix(n):
    """
    Returns a block diagonal matrix with all one matrix on each block.
    """

    A_1 = np.ones((int(n/3), int(n/3)))
    A_2 = np.ones((int(n/3), int(n/3)))
    A_3 = np.ones((int(n/3), int(n/3)))
    A = sp.linalg.block_diag(A_1, A_2, A_3)
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
    print("The local minimum is: {}\n".format(opt_val))
    print("The corresponding minimizer is: \n{}\n".format(X))

    return theta, X


def gradient_norm_func(A, theta):

    return np.linalg.norm(trig.trig_grad(A, theta))


if __name__ == '__main__':
    n = 12
    sigma = 0.0001

    B = build_block_matrix(n)
    extra = sigma * np.ones((n, n))
    print("The block matrix B:\n{}\n".format(B))

    theta_1 = np.zeros(int(n/3))
    theta_2 = np.random.uniform(0, 2 * np.pi) * np.ones(int(n/3))
    theta_3 = np.random.uniform(0, 2 * np.pi) * np.ones(int(n/3))
    theta = np.hstack((theta_1, theta_2, theta_3))
    # theta = np.zeros(n)
    print("The spurious solution on B:\n{}\n".format(theta))
    # dim, _ = B.shape
    # T = np.empty((dim, dim))
    # for i in range(dim):
    #     T[i, :] = theta[i]
    #     T[i, :] = T[i, :] - theta
    # cosT = np.cos(T)
    # sinT = np.sin(T)

    S, theta_sp = trig.trig_bfgs(B, None, init=None)
    print("The spurious solution:\n{}\n".format(S))
    print("The spurious solution in trig:\n{}\n".format(np.mod(theta_sp, 2 * np.pi)))

    grad = trig.trig_grad(B, theta)
    print("The gradient of function at spurious solution:\n{}\n".format(grad))
    hess = trig.trig_hess(B, theta)
    eigs = aux.sorted_eigenvalues(hess).ravel()
    print("The eigs of hessian of function at spurious solution:\n{}\n".format(eigs))

    A = (B + extra) / sigma
    A = A - np.diag(np.diag(A))
    # print("The observation looks like block matrix:\n{}\n".format(A))

    # grad_extra = trig.trig_grad(A, theta)
    # print("The gradient of function at spurious solution:\n{}\n".format(grad_extra))
    # hess_extra = trig.trig_hess(A, theta)
    # eigs_extra = aux.sorted_eigenvalues(hess_extra).ravel()
    # print("The eigs of hessian of function at spurious solution:\n{}\n".format(eigs_extra))
    #
    # theta_out, X = minimize_gradient(A, theta)
    # print("Optimization result using min gradient:\n{}\n".format(theta_out))

    grad = trig.trig_grad(A, theta_sp)
    print("The gradient of true function at spurious solution:\n{}\n".format(grad))
    hess = trig.trig_hess(A, theta_sp)
    eigs = aux.sorted_eigenvalues(hess).ravel()
    print("The eigs of hessian of true function at spurious solution:\n{}\n".format(eigs))

    Q, theta_local = trig.trig_bfgs(A, None, init=theta_sp)
    print("Optimization result using min function value:\n{}\n".format(np.mod(theta_local, 2 * np.pi)))
    # dim, _ = B.shape
    # T = np.empty((dim, dim))
    # for i in range(dim):
    #     T[i, :] = theta_local[i]
    #     T[i, :] = T[i, :] - theta_local
    # cosT = np.cos(T)
    # sinT = np.sin(T)
    # print(T)
    # print(cosT)

    # grad_trig = trig.trig_grad(A, theta_local)
    # print("The gradient at this point is:\n{}\n".format(grad_trig))
    # hess_trig = trig.trig_hess(A, theta_local)
    # print("The eigenvalue of the corresponding Hessian:\n{}\n".format(aux.sorted_eigenvalues(hess_trig).ravel()))

    # Q = bm.augmented_lagrangian(A, 2, plotting=False, printing=True, init=S)
    result = bm.trust_region(A, 2, init=S)
    Q = bm._vector_to_matrix(result.x, 2)
    print("The place where spurious solution converges:\n{}\n".format(Q))
    print("The gradient at this point:\n{}\n".format(A.dot(Q)))
    hess_Q = A * (Q.dot(Q.T)) - np.diag(np.sum(A * Q.dot(Q.T), axis=1))
    eigs = aux.sorted_eigenvalues(hess_Q).ravel()
    print("The eigenvalue of the corresponding Hessian:\n{}\n".format(eigs))
