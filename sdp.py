import numpy as np
import cvxpy as cvx
import sync_generator as gen
import aux


def sdp_relaxation(Y, z):
    print('Solving sdp relaxation problem...')
    n, _ = Y.shape
    X = cvx.Semidef(n)
    A = X * Y
    objective = cvx.Maximize(cvx.trace(A))
    constraints = [cvx.diag(X) == 1]
    problem = cvx.Problem(objective, constraints)
    problem.solve()
    print('Status: ' + problem.status)
    print('Optimal value: \n', problem.value)
    print('Verifying optimality (dual value): \n',
          np.sum(constraints[0].dual_value))
    print('Optimal X: \n', X.value)
    print('Optimal dual D (only diagonal entries): \n',
          constraints[0].dual_value)
    return problem.value, X, constraints[0].dual_value


if __name__ == "__main__":
    Y, z = gen.synchronization_usual(1000, .5, 10)
    value, X, _ = sdp_relaxation(Y, z)
    Z = z.dot(z.T)
    print(np.linalg.norm(Z - X))
