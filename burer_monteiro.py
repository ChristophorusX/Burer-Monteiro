from __future__ import division, print_function, absolute_import
import math
import scipy.linalg
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import sbm_generator as gensbm
import sync_generator as gensync
import aux
from matplotlib import rc
rc('text', usetex=True)

def augmented_lagrangian(Y, k, plotting=False, printing=True):
    """Returns the resulting local minimizer R of the BM problem."""

    n, _ = Y.shape
    y = np.ones(n).reshape((-1, 1))
    R = np.random.uniform(-1, 1, size=(n, k))
    penalty = 1
    gamma = 10
    eta = .25
    target = .01
    vec = _constraint_term_vec(n, R)
    v = vec.reshape((1, -1)).dot(vec)
    v_best = v
    while v > target:
        Rv = _matrix_to_vector(R)
        if printing == True:
            print('Starting L-BFGS-B on augmented Lagrangian...')
        optimizer = opt.minimize(lambda R_vec: _augmented_lagrangian_func(
            R_vec, Y, y, penalty, n, k), Rv, jac=lambda R_vec: _jacobian(R_vec, Y, n, y, penalty, k), method="L-BFGS-B")
        if printing == True:
            print('Finishing L-BFGS-B on augmented Lagrangian...')
        R = _vector_to_matrix(optimizer.x, k)
        vec = _constraint_term_vec(n, R)
        v = vec.reshape((1, -1)).dot(vec)
        if printing == True:
            print('Finish updating variables...')
        if plotting == True:
            _plot_R(R)
        if v < eta * v_best:
            y = y - penalty * vec
            v_best = v
        else:
            penalty = gamma * penalty
    if printing == True:
        print('Augmented Lagrangian terminated.')
    return R


def _generate_random_rect(n, k):
    """Returns a random initialization of matrix."""

    R = np.random.uniform(-1, 1, (n, k))
    for i in range(n):
        R[i, :] = R[i, :] / np.linalg.norm(R[i, :])
    return R


def _basis_vector(size, index):
    """Returns a basis vector with 1 on certain index."""

    vec = np.zeros(size)
    vec[index] = 1
    return vec


def _A_trace_vec(n, R):
    """Returns a vector containing norm square of row vectors of R."""

    vec = np.empty(n)
    for i in range(n):
        vec[i] = R[i, :].dot(R[i, :])
    return vec.reshape((-1, 1))


def _constraint_term_vec(n, R):
    """Returns the vector required to compute objective function value."""

    vec = _A_trace_vec(n, R)
    constraint = vec - np.ones(n).reshape((-1, 1))
    return constraint


def _augmented_lagrangian_func(Rv, Y, y, penalty, n, k):
    """Returns the value of objective function of augmented Lagrangian."""

    R = _vector_to_matrix(Rv, k)
    vec = _constraint_term_vec(n, R)
    objective = -np.trace(Y.dot(R.dot(R.T))) - y.reshape((1, -1)
                    ).dot(vec) + penalty / 2 * vec.reshape((1, -1)).dot(vec)
    return objective


def _vector_to_matrix(Rv, k):
    """Returns a matrix from reforming a vector."""
    U = Rv.reshape((-1, k))
    return U


def _matrix_to_vector(R):
    """Returns a vector from flattening a matrix."""

    u = R.reshape((1, -1)).ravel()
    return u

# def _take_one_row(Z, R, l):
#     Z[l,:] = R[l,:]
#     return Z


def _jacobian(Rv, Y, n, y, penalty, k):
    """Returns the Jacobian matrix of the augmented Lagrangian problem."""

    R = _vector_to_matrix(Rv, k)
    vec_trace_A = _A_trace_vec(n, R).ravel()
    vec_second_part = R.copy()
    for l in range(n):
        vec_second_part[l, :] *= y.ravel()[l]
    vec_third_part = R.copy()
    for l in range(n):
        vec_third_part[l, :] *= (vec_trace_A[l] - 1)
    jacobian = -2 * Y.dot(R) - 2 * vec_second_part + \
        2 * penalty * vec_third_part
    jac_vec = _matrix_to_vector(jacobian)
    return jac_vec.reshape((1, -1)).ravel()


def _plot_R(R):
    """Plot the found matrices R on their row vectors."""

    plt.style.use('ggplot')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.scatter(R.T[0], R.T[1], alpha=0.25, label='Rows of R')
    plt.legend()
    plt.show()


def trust_region(A, k, plotting=False, printing=False):
    """Returns a Result object containing information of the local minimizer."""

    print('Starting trust region on manifold...')
    n, _ = A.shape
    Y = _generate_random_rect(n, k)
    return minimize_with_trust(lambda Yv: obj_function(A, Yv, n, k), Y, k, plotting, printing, jac=lambda Yv: _proj_grad_from_vec(
        A, Yv, n, k), hessp=lambda Yv, Tv: _hessian_p(A, Yv, Tv, n, k))


def trust_region_plotting(A, k):
    """Same as function trust_region, with an extra plotting functionality."""

    print('Starting trust region on manifold...')
    n, _ = A.shape
    Y = _generate_random_rect(n, k)
    return minimize_with_trust(lambda Yv: obj_function(A, Yv, n, k), Y, k, jac=lambda Yv: _proj_grad_from_vec(
        A, Yv, n, k), hessp=lambda Yv, Tv: _hessian_p(A, Yv, Tv, n, k))


def obj_function(A, Yv, n, k):
    """Returns objective function for the trust region method"""

    Y = _vector_to_matrix(Yv, k)
    return -np.trace(A.dot(Y.dot(Y.T)))


def _projection(Z, Y):
    """Returns projected point from tangent plane back to the manifold."""

    dia = np.diag(np.diag(Z.dot(Y.T)))
    return Z - dia.dot(Y)


def _grad(A, Y):
    """Returns gradient of the objective function."""

    return -A.dot(Y)


def _proj_grad_from_vec(A, Yv, n, k):
    """Returns the projected gradient given point on manifold in vector form."""

    Y = _vector_to_matrix(Yv, k)
    proj_grad = _projection(_grad(A, Y), Y)
    return _matrix_to_vector(proj_grad)


def _hessian_p(A, Yv, Tv, n, k):
    """Returns the directional Hessian matrix."""

    Y = _vector_to_matrix(Yv, k)
    T = _vector_to_matrix(Tv, k)
    directional_hess = (A - np.diag((A.dot(Y)).dot(Y.T))).dot(T)
    return _matrix_to_vector(directional_hess)


def _retraction(Tv):
    """Returns the retracted point given one on the tangent plane."""

    T = _vector_to_matrix(Tv, 2)
    n, _ = T.shape
    for i in range(n):
        T[i, :] = T[i, :] / np.linalg.norm(T[i, :])
    return _matrix_to_vector(T)


"""Trust-region optimization."""


def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in Scipy. Level 4 is the first
        # level in user code.
        # warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)


def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper


_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.'}


class Result(dict):
    """ Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess, hess_inv : ndarray
        Values of objective function, Jacobian, Hessian or its inverse (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"


__all__ = []


class BaseQuadraticSubproblem(object):
    """
    Base/abstract class defining the quadratic model for trust-region
    minimization. Child classes must implement the ``solve`` method.

    Values of the objective function, jacobian and hessian (if provided) at
    the current iterate ``x`` are evaluated on demand and then stored as
    attributes ``fun``, ``jac``, ``hess``.
    """

    def __init__(self, x, fun, jac, hess=None, hessp=None):
        self._x = x
        self._f = None
        self._g = None
        self._h = None
        self._g_mag = None
        self._cauchy_point = None
        self._newton_point = None
        self._fun = fun
        self._jac = jac
        self._hess = hess
        self._hessp = hessp

    def __call__(self, p):
        return self.fun + np.dot(self.jac, p) + 0.5 * np.dot(p, self.hessp(p))

    @property
    def fun(self):
        """Value of objective function at current iteration."""
        if self._f is None:
            self._f = self._fun(self._x)
        return self._f

    @property
    def jac(self):
        """Value of jacobian of objective function at current iteration."""
        if self._g is None:
            self._g = self._jac(self._x)
        return self._g

    @property
    def hess(self):
        """Value of hessian of objective function at current iteration."""
        if self._h is None:
            self._h = self._hess(self._x)
        return self._h

    def hessp(self, p):
        if self._hessp is not None:
            return self._hessp(self._x, p)
        else:
            return np.dot(self.hess, p)

    @property
    def jac_mag(self):
        """Magniture of jacobian of objective function at current iteration."""
        if self._g_mag is None:
            self._g_mag = scipy.linalg.norm(self.jac)
        return self._g_mag

    def get_boundaries_intersections(self, z, d, trust_radius):
        """
        Solve the scalar quadratic equation ||z + t d|| == trust_radius.
        This is like a line-sphere intersection.
        Return the two values of t, sorted from low to high.
        """
        a = np.dot(d, d)
        b = 2 * np.dot(z, d)
        c = np.dot(z, z) - trust_radius**2
        sqrt_discriminant = math.sqrt(b * b - 4 * a * c)
        ta = (-b - sqrt_discriminant) / (2 * a)
        tb = (-b + sqrt_discriminant) / (2 * a)
        return ta, tb

    def solve(self, trust_radius):
        raise NotImplementedError('The solve method should be implemented by '
                                  'the child class')


def _minimize_trust_region(fun, x0, n_rows, plotting, printing, args=(), jac=None, hess=None, hessp=None,
                           subproblem=None, initial_trust_radius=1.0,
                           max_trust_radius=1000.0, eta=0.15, gtol=1e-4,
                           maxiter=None, disp=False, return_all=False,
                           callback=None, **unknown_options):
    """
    Minimization of scalar function of one or more variables using a
    trust-region algorithm.

    Options for the trust-region algorithm are:
        initial_trust_radius : float
            Initial trust radius.
        max_trust_radius : float
            Never propose steps that are longer than this value.
        eta : float
            Trust region related acceptance stringency for proposed steps.
        gtol : float
            Gradient norm must be less than `gtol`
            before successful termination.
        maxiter : int
            Maximum number of iterations to perform.
        disp : bool
            If True, print convergence message.

    This function is called by the `minimize` function.
    It is not supposed to be called directly.
    """
    _check_unknown_options(unknown_options)

    # force the initial guess into a nice format
    x0 = np.asarray(x0).flatten()

    # Wrap the functions, for a couple reasons.
    # This tracks how many times they have been called
    # and it automatically passes the args.
    nfun, fun = wrap_function(fun, args)
    njac, jac = wrap_function(jac, args)
    nhess, hess = wrap_function(hess, args)
    nhessp, hessp = wrap_function(hessp, args)

    # limit the number of iterations
    if maxiter is None:
        maxiter = len(x0) * 200

    # init the search status
    warnflag = 0

    # initialize the search
    trust_radius = initial_trust_radius
    x = x0
    if return_all:
        allvecs = [x]
    m = subproblem(x, fun, jac, hess, hessp)
    k = 0

    # search for the function min
    while True:

        # Solve the sub-problem.
        # This gives us the proposed step relative to the current position
        # and it tells us whether the proposed step
        # has reached the trust region boundary or not.
        try:
            p, hits_boundary = m.solve(trust_radius)
        except np.linalg.linalg.LinAlgError as e:
            warnflag = 3
            break

        # calculate the predicted value at the proposed point
        predicted_value = m(p)

        # define the local approximation at the proposed point
        # retract the point x from tangent space to the manifold
        if printing == True:
            print('Start retracting onto the manifold...')
        x_proposed = _retraction(x + p)
        m_proposed = subproblem(x_proposed, fun, jac, hess, hessp)

        # evaluate the ratio defined in equation (4.4)
        actual_reduction = m.fun - m_proposed.fun
        predicted_reduction = m.fun - predicted_value
        # if predicted_reduction <= 0:
        #     warnflag = 2
        #     break
        rho = actual_reduction / predicted_reduction

        # update the trust radius according to the actual/predicted ratio
        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2 * trust_radius, max_trust_radius)

        # if the ratio is high enough then accept the proposed step
        if rho > eta:
            x = x_proposed
            m = m_proposed
            if printing == True:
                print('Proposed step accepted...')
            R = _vector_to_matrix(x, n_rows)
            if plotting == True:
                _plot_R(R)

        # append the best guess, call back, increment the iteration count
        if return_all:
            allvecs.append(x)
        if callback is not None:
            callback(x)
        k += 1

        # check if the gradient is small enough to stop
        if m.jac_mag < gtol:
            warnflag = 0
            break

        # check if we have looked at enough iterations
        if k >= maxiter:
            warnflag = 1
            break

    # print some stuff if requested
    status_messages = (
        _status_message['success'],
        _status_message['maxiter'],
        'A bad approximation caused failure to predict improvement.',
        'A linalg error occurred, such as a non-psd Hessian.',
    )
    if warnflag == 0:
        print(status_messages[warnflag])
    else:
        print('Warning: ' + status_messages[warnflag])
    # print("         Current function value: %f" % m())
    print("         Iterations: %d" % k)
    print("         Function evaluations: %d" % nfun[0])
    print("         Gradient evaluations: %d" % njac[0])
    print("         Hessian evaluations: %d" % nhess[0])

    result = Result(x=x, success=(warnflag == 0), status=warnflag, fun=m.fun,
                    jac=m.jac, nfev=nfun[0], njev=njac[0], nhev=nhess[0],
                    nit=k, message=status_messages[warnflag])

    if hess is not None:
        result['hess'] = m.hess

    if return_all:
        result['allvecs'] = allvecs

    return result


"""Newton-CG trust-region optimization."""


def _minimize_trust_ncg(fun, x0, n_rows, plotting, printing, args=(), jac=None, hess=None, hessp=None,
                        **trust_region_options):
    return _minimize_trust_region(fun, x0, n_rows, plotting, printing, args=args, jac=jac, hess=hess,
                                  hessp=hessp, subproblem=CGSteihaugSubproblem,
                                  **trust_region_options)


class CGSteihaugSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by a conjugate gradient method"""

    def solve(self, trust_radius):
        """
        Solve the subproblem using a conjugate gradient method.

        Parameters
        ----------
        trust_radius : float
            We are allowed to wander only this far away from the origin.

        Returns
        -------
        p : ndarray
            The proposed step.
        hits_boundary : bool
            True if the proposed step is on the boundary of the trust region.

        Notes
        -----
        This is algorithm (7.2) of Nocedal and Wright 2nd edition.
        Only the function that computes the Hessian-vector product is required.
        The Hessian itself is not required, and the Hessian does
        not need to be positive semidefinite.
        """

        # get the norm of jacobian and define the origin
        p_origin = np.zeros_like(self.jac)

        # define a default tolerance
        tolerance = min(0.5, math.sqrt(self.jac_mag)) * self.jac_mag

        # Stop the method if the search direction
        # is a direction of nonpositive curvature.
        if self.jac_mag < tolerance:
            hits_boundary = False
            return p_origin, hits_boundary

        # init the state for the first iteration
        z = p_origin
        r = self.jac
        d = -r

        # Search for the min of the approximation of the objective function.
        while True:

            # do an iteration
            Bd = self.hessp(d)
            dBd = np.dot(d, Bd)
            if dBd <= 0:
                # Look at the two boundary points.
                # Find both values of t to get the boundary points such that
                # ||z + t d|| == trust_radius
                # and then choose the one with the predicted min value.
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                pa = z + ta * d
                pb = z + tb * d
                if self(pa) < self(pb):
                    p_boundary = pa
                else:
                    p_boundary = pb
                hits_boundary = True
                return p_boundary, hits_boundary
            r_squared = np.dot(r, r)
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if scipy.linalg.norm(z_next) >= trust_radius:
                # Find t >= 0 to get the boundary point such that
                # ||z + t d|| == trust_radius
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                p_boundary = z + tb * d
                hits_boundary = True
                return p_boundary, hits_boundary
            r_next = r + alpha * Bd
            r_next_squared = np.dot(r_next, r_next)
            if math.sqrt(r_next_squared) < tolerance:
                hits_boundary = False
                return z_next, hits_boundary
            beta_next = r_next_squared / r_squared
            d_next = -r_next + beta_next * d

            # update the state for the next iteration
            z = z_next
            r = r_next
            d = d_next


def minimize_with_trust(fun, x0, n_rows, plotting, printing, args=(), jac=None, hess=None,
                        hessp=None, callback=None, options=None):
    if options is None:
        options = {}
    return _minimize_trust_ncg(fun, x0, n_rows, plotting, printing, args, jac, hess, hessp,
                               callback=callback, **options)


if __name__ == "__main__":
    # Y, z = gensbm.sbm_linear(500, 10, 2)
    # Y = aux.demean(Y, 10, 2)
    Y, z = gensync.synchronization_usual(1000, .5, 10)
    n, _ = Y.shape
    # print(Y)
    # print(Y)
    # Y_re = Y.reshape((1, -1)).ravel()
    # print(Y_re)
    # Y_back = Y_re.reshape((10,10))
    # print(Y_back)
    augmented_lagrangian(Y, 2, plotting=True)
    # result = trust_region(Y, 2)
