import spectral_gap_analysis as sp
import burer_monteiro as bm
import numpy as np
from matplotlib import pyplot as plt


def landscape(A, z, max_radius):
    """Compute differences between function values and prediction values of some
       random samples on manifold in the certain domain.
    """

    info = []
    radius_range = np.linspace(
        0, max_radius, num=5000, endpoint=True, retstep=False)
    for radius in radius_range:
        rand_ground = get_ground_truth(z)
        pt = get_nearby_pt(rand_ground, radius)
        func_value = func_val(A, pt)
        pred_value = pred_val(A, rand_ground, pt)
        diff = func_value - pred_value
        info.append([diff, radius])
    return np.array(info)


def func_val(A, point):
    """Return function value at the point."""

    return np.trace((A.dot(point)).dot(point.T))


def pred_val(A, ground_truth, point):
    """Return the predicted upper bound given the strong concavity of the function."""

    dim = A.shape[0]
    grad = A.dot(ground_truth)
    distance = point - ground_truth
    return func_val(A, ground_truth) + np.sum(grad * distance) + dim / 2 * np.linalg.norm(distance)**2


def draw_landscape(info):
    """Draw the difference against radius."""

    diff_arr = info[:, 0]
    radius_arr = info[:, 1]
    plt.style.use('ggplot')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.scatter(radius_arr, diff_arr, alpha=0.75, label=r'Point $Q\in\mathcal{{{M}}}$ near ground truth')
    plt.title(r'Local landscape near ground truth\\ compared to ``quadratic" approximation')
    plt.xlabel(r'Radius $r$ to ground truth')
    plt.ylabel(r'Difference $\mathrm{{{Tr}}}(AQ) -\mathrm{{{model}}}(Q)$')
    plt.text(0, -10, r'$\mathrm{{{model}}}(Q)=f(Q(z))+\langle\mathrm{{{grad}}}f(Q(z)),Q-Q(z)\rangle+\frac{{{n}}}{{{2}}}\|Q-Q(z)\|^2$')
    plt.legend()
    plt.savefig('local landscape.png', dpi=250)
    plt.close('all')


def get_observation(n, level, noise_type):
    """Obtain an observation and ground truth from certain type of perturbation."""

    if noise_type == 'positive':
        z = np.ones(n).reshape((-1, 1))
        ground_truth = z.dot(z.T)
        N = sp._gen_row_mat(n, level)
        N = N - np.diag(np.diag(N))
        A = ground_truth + N
    return A, z


def get_ground_truth(z):
    """Get a corresponding matrix on the manifold from given ground truth."""

    dim = z.ravel().shape[0]
    z = z.reshape((-1, 1))
    zero = np.zeros(dim).reshape((-1, 1))
    mat = np.hstack((z, zero))
    ground_truth = mat.dot(_gen_orthogonal())
    return ground_truth


def _gen_orthogonal(dim=2):
    """Generate an orthogonal matrix of a certain dimension."""

    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


def get_nearby_pt(ground_truth, distance):
    """Get a nearby point of certain distance given the position of ground truth."""

    dim = ground_truth.shape[0]
    # generate a vector on tangent plane as a direction
    tangent = np.hstack((ground_truth[:, 1].reshape((-1, 1)), ground_truth[:, 0].reshape((-1, 1))))
    # create a direction orthogonal to the one above (that gives 0 eigenvalue)
    direction = np.random.random_sample(dim)
    direction = direction - np.mean(direction)
    direction = direction / np.linalg.norm(direction)
    add_on = np.diag(direction).dot(tangent)
    # direction times distance requested
    add_on = distance * add_on
    point = ground_truth + add_on
    # project back onto the manifold
    for i in range(dim):
        point[i, :] = point[i, :] / np.linalg.norm(point[i, :])
    return point


if __name__ == '__main__':
    A, z = get_observation(10, 10, 'positive')
    info = landscape(A, z, max_radius=.5)
    draw_landscape(info)
