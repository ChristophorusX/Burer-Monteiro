import matplotlib
matplotlib.use('Agg')
import spectral_gap_analysis as sp
import numpy as np
from matplotlib import pyplot as plt
import noise_generator as gen
import sync_generator as sync
import burer_monteiro as bm


def landscape(A, z, max_radius):
    """
    Compute differences between function values and prediction values of some
    random samples on manifold in the certain domain.
    """

    info = []
    dim, _ = A.shape
    radius_range = np.linspace(
        0, max_radius, num=5000, endpoint=True, retstep=False)
    for radius in radius_range:
        rand_ground = get_ground_truth(z)
        pt, distance = get_nearby_pt(rand_ground, radius)
        func_value = func_val(A, pt)
        pred_value = pred_val(A, rand_ground, pt)
        diff = func_value - pred_value
        correlation = np.linalg.norm(z.ravel().dot(pt)) / dim
        info.append([diff, distance, correlation])
    return np.array(info)


def correlation_landscape(A, z, loops):
    """
    Compute the trajectory of correlations of the points on manifold along the
    descendence of trust region algorithm.
    """

    correlation_arr_array = []
    for i in range(loops):
        correlation_arr = []
        result = bm.trust_region(A, 2, plotting=False, printing=False, correlation_arr=correlation_arr, ground_truth=z)
        Q = bm._vector_to_matrix(result.x, 2)
        QT = Q.transpose()
        correlation = np.linalg.norm(QT.dot(z)) / z.shape[0]
        if correlation > .95:
            correlation_arr = np.array(correlation_arr)
            correlation_arr_array.append(correlation_arr)
    return correlation_arr_array


def func_val(A, point):
    """Return function value at the point."""

    return np.trace((A.dot(point)).dot(point.T))


def pred_val(A, ground_truth, point):
    """Return the predicted upper bound given the strong concavity of the function."""

    dim = A.shape[0]
    grad = A.dot(ground_truth)
    distance = point - ground_truth
    return func_val(A, ground_truth) + np.sum(grad * distance) - dim / 2 * np.linalg.norm(distance)**2


def draw_landscape(info):
    """Draw the difference against distance."""

    diff_arr = info[:, 0]
    distance_arr = info[:, 1]
    corr_arr = info[:, 2]
    plt.style.use('ggplot')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.scatter(distance_arr, diff_arr, alpha=0.5,
                label=r'Point $Q\in\mathcal{M}$ near ground truth (color indicates correlation)',
                c=corr_arr, cmap=plt.cm.get_cmap('inferno'), vmin=.5, vmax=1)
    plt.title(
        r'Local landscape near ground truth\\ compared to ``quadratic" approximation')
    plt.xlabel(r'Distance $\|zz^T-QQ^T\|_F$ to ground truth')
    plt.ylabel(r'Difference $\mathrm{Tr}(AQ) -\mathrm{model}(Q)$')
    # plt.text(
    #     0, -8, r'$\mathrm{model}(Q)=f(\bar Q(z))+\langle\mathrm{grad}f(\bar Q(z)),Q-\bar Q(z)\rangle-\frac{n}{2}\|Q-\bar Q(z)\|^2$')
    plt.legend()
    plt.savefig('local landscape.png', dpi=250)
    plt.close('all')


def draw_correlation_landscape(correlation_arr_array):
    """Draw the trajectories of correlation change with different initializations."""

    plt.style.use('ggplot')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    for arr in correlation_arr_array:
        plt.plot(range(arr.shape[0]), arr, alpha=.5)
    plt.title('Correlation Trajectories under Trust Region')
    plt.xlabel(r'Number of iterations $k$')
    plt.ylabel(r'Correlation $\|Q^Tz\|_2/n$')
    plt.savefig('correlation trajectories.png', dpi=250)
    plt.close('all')


def get_observation(n, level, noise_type):
    """Obtain an observation and ground truth from certain type of perturbation."""

    if noise_type == 'positive-rows':
        z = np.ones(n).reshape((-1, 1))
        ground_truth = z.dot(z.T)
        N = sp._gen_row_mat(n, level)
        N = N - np.diag(np.diag(N))
        A = ground_truth + N
    elif noise_type == 'positive-sparse':
        z = np.ones(n).reshape((-1, 1))
        ground_truth = z.dot(z.T)
        N = sp._gen_sparse_mat(n, level)
        N = N - np.diag(np.diag(N))
        A = ground_truth + N
    elif noise_type == 'positive-uniform':
        z = np.ones(n).reshape((-1, 1))
        ground_truth = z.dot(z.T)
        N = gen.uniform_noise(n, level) + level
        N = N - np.diag(np.diag(N))
        A = ground_truth + N
    elif noise_type == 'uniform':
        z = np.ones(n).reshape((-1, 1))
        ground_truth = z.dot(z.T)
        N = gen.uniform_noise(n, level)
        N = N - np.diag(np.diag(N))
        A = ground_truth + N
    elif noise_type == 'sync':
        snr = level
        A, z = sync.synchronization_usual(n, .5, snr)

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
    tangent = np.hstack((ground_truth[:, 1].reshape(
        (-1, 1)), ground_truth[:, 0].reshape((-1, 1))))
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
    distance = np.linalg.norm(ground_truth.dot(ground_truth.T) - point.dot(point.T))
    return point, distance


if __name__ == '__main__':
    A, z = get_observation(200, 10, 'positive-sparse')
    # info = landscape(A, z, max_radius=500)
    # draw_landscape(info)
    correlation_arr_array = correlation_landscape(A, z, 200)
    draw_correlation_landscape(correlation_arr_array)
