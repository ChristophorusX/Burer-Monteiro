import numpy as np
import basin_of_attraction as basin
import stationary_points as stat
from scipy import optimize as opt


def ground_truth_rotation(ground_truth, current_point):
    """
    Returns the found orthogonal matrix R that minimizes the distance between
    the given point and ground truth in Frobenius norm.
    """

    dim = ground_truth.ravel().shape[0]
    zeros = np.zeros(dim).reshape((-1, 1))
    ground_truth = ground_truth.reshape((-1, 1))
    Q_star = np.hstack((ground_truth, zeros))
    init = basin._gen_orthogonal(dim=2).ravel()
    result = opt.minimize(
        fun=lambda R_vec: np.linalg.norm(Q_star.dot(R_vec.reshape(2, 2)) - current_point), x0=init)
    R_vec = result.x
    opt_val = result.fun
    return R_vec.reshape((2, 2)), opt_val


def form_direction(ground_truth, current_point, rotation):
    """
    Returns the direction from the best ground truth up to rotation to the
    current point.
    """

    dim = ground_truth.ravel().shape[0]
    zeros = np.zeros(dim).reshape((-1, 1))
    ground_truth = ground_truth.reshape((-1, 1))
    Q_star = np.hstack((ground_truth, zeros))

    return current_point - Q_star.dot(rotation)


def evaluate_hessian_val(A, point, direction):
    """
    Returns the value of Hessian function in the given direction.
    """

    hess_p = (A - np.diag((A.dot(point)).dot(point.T))).dot(direction)
    return np.sum(hess_p * direction)


if __name__ == '__main__':
    while True:
        A, z = basin.get_observation(10, 3, 'sync')
        z = z.reshape((-1, 1))
        dim, _ = A.shape

        for i in range(1000):
            current_point = stat.full_circle_rect(dim)

            rotation, _ = ground_truth_rotation(z, current_point)
            direction = form_direction(z, current_point, rotation)
            val = evaluate_hessian_val(A, current_point, direction)
            print("The curvature evaluation at the direction: {}".format(val))
