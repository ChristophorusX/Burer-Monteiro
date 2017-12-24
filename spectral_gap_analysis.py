import numpy as np
import sync_generator as syncgen
import sbm_generator as sbmgen
import noise_generator as gen
import burer_monteiro as bm
from sklearn import cluster
import aux


def _spectral_gap(A, z):
    gap = aux.laplacian_eigs(A, z)[1]
    print('Spectral gap: {}'.format(gap))
    return gap


def _gen_sync(n, percentage, snr):
    return syncgen.synchronization_usual(n, percentage, snr)


def _gen_sbm(n, a, b):
    return sbmgen.sbm_logarithm(n, a, b)


def _check_spectral_gap(A, z):
    if _spectral_gap(A, z) > .001:
        return True
    else:
        return False


def _gen_sparse_mat(n, level):
    mat = np.zeros(n**2)
    for i in range(2):
        index = np.random.randint(n**2)
        mat[index] = level
    mat = mat.reshape((n, n))
    return mat


def _gen_row_mat(n, level):
    mat = np.zeros((n, n))
    for i in range(2):
        index = np.random.randint(n)
        mat[index, :] = level
        mat[:, index] = level
    return mat


def search_counter_eg(n, level, drift, n_iter, n_trail):
    # found_target = False
    examples = []

    while True:  # level > 0
        # level += .05
        print('+++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Starting loops with noise level = {}...'.format(level))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++')

        n_tests = 0
        print('This is #{} loop...........'.format(n_tests))

        for i in range(n_iter):
            print('Loop #{}'.format(i + 1))
            # z = aux.rounding_with_prob(np.random.random_sample(n), .5)
            # z = 2 * z.reshape(n, 1) - 1
            z = np.ones(n).reshape((-1, 1))
            ground_truth = z.dot(z.T)
            # noise = [i for i in range(n)]
            # noise = np.array(noise).reshape((-1, 1))
            # mat = noise.dot(noise.T)
            # N_pre = aux.laplacian(mat)
            # N_pre = N_pre - np.diag(np.diag(N_pre))
            # N = - level * N_pre

            # N = gen.uniform_noise(n, level) + drift
            N = _gen_row_mat(n, level)
            N = N - np.diag(np.diag(N))
            A = ground_truth + N
            # A = aux.demean_adversary(A)
            # A, z = _gen_sbm(n, 10, 2)

            if _check_spectral_gap(A, z):
                print('------------Found matrix where SDP tight------------')
                for j in range(n_trail):
                    print(
                        'Finding global optimizer with BM (trail {})...'.format(j + 1))
                    # Q = bm.augmented_lagrangian(
                    #     A, 2, plotting=False, printing=False)
                    result = bm.trust_region(A, 2, plotting=False)
                    Q_vec = result.x
                    Q = Q_vec.reshape((n, 2))
                    flag = result.success
                    print('Success: {}'.format(flag))
                    # kmeans = cluster.KMeans(
                    #     n_clusters=2, random_state=0).fit(Q)
                    # clustering = 2 * kmeans.labels_ - 1
                    # err = aux.error_rate(clustering, z.ravel())
                    # print('The error rate for BM is: {}...'.format(err))
                    X_result = Q.dot(Q.T)
                    X = z.dot(z.T)
                    err = np.linalg.norm(X - X_result, 1)
                    corr = np.linalg.norm(np.dot(Q.T, z), 2)
                    largest_diff = np.max(np.abs(X - X_result))
                    pair_diff = 0
                    for k in range(n):
                        for l in range(n):
                            if k != l:
                                vec1 = Q[k, :]
                                vec2 = Q[l, :]
                                d = np.linalg.norm(vec1 - vec2, 2)
                                if d > pair_diff:
                                    pair_diff = d

                    print('>>>>>>The correlation factor is: {}...'.format(corr / n))
                    print('>>>>>>The norm 1 error for BM is: {}...'.format(err / n**2))
                    print('>>>>>>The largest element diff is: {}...'.format(
                        largest_diff))
                    print('>>>>>>The largest pairwise difference is: {}...'.format(
                        pair_diff))
                    N = A - z.dot(z.T)
                    diagN = np.diag(N.dot(z).ravel())
                    spectral_overall = np.sort(np.linalg.eigvals(N - diagN))
                    print('Max eigenvalue overall: {}'.format(
                        spectral_overall[-1]))
                    spectral_N = np.sort(np.linalg.eigvals(N))
                    print('###### Max eigenvalue of N: {} ######'.format(
                        spectral_N[-1]))
                    print('Min eigenvalue of N: {}'.format(spectral_N[0]))
                    spectral_diagN = np.sort(np.linalg.eigvals(diagN))
                    print('Max eigenvalue of diagN: {}'.format(
                        spectral_diagN[-1]))
                    print('###### Min eigenvalue of diagN: {} ######'.format(
                        spectral_diagN[0]))
                    if pair_diff > .1:
                        gap = aux.laplacian_eigs(A, z)[1]
                        if gap > .01:
                            # found_target = True
                            print(
                                'One instance found when noise level = {}!'.format(level))
                            example = CounterExample(A, z, Q, gap, level)
                            examples.append(example)
                            print(A)
                            exit(0)
            else:
                print('===SDP fails===')
                Q = bm.augmented_lagrangian(
                    A, 2, plotting=False, printing=False)
                kmeans = cluster.KMeans(
                    n_clusters=2, random_state=0).fit(Q)
                clustering = 2 * kmeans.labels_ - 1
                err = aux.error_rate(clustering, z.ravel())
                print('===Error rate for BM is: {}==='.format(err))
                Q = bm.augmented_lagrangian(
                    A, 2, plotting=False, printing=False)
                # kmeans = cluster.KMeans(
                #     n_clusters=2, random_state=0).fit(Q)
                # clustering = 2 * kmeans.labels_ - 1
                # err = aux.error_rate(clustering, z.ravel())
                # print('The error rate for BM is: {}...'.format(err))
                X_result = Q.dot(Q.T)
                X = z.dot(z.T)
                err = np.linalg.norm(X - X_result, 1)
                corr = np.linalg.norm(np.dot(Q.T, z), 2)
                largest_diff = np.max(np.abs(X - X_result))
                print('The correlation factor is: {}...'.format(corr / n))
                print('The norm 1 error for BM is: {}...'.format(err / n**2))
                print('The largest element diff is: {}...'.format(largest_diff))
                N = A - z.dot(z.T)
                diagN = np.diag(N.dot(z).ravel())
                spectral_overall = np.sort(np.linalg.eigvals(N - diagN))
                print('Max eigenvalue overall: {}'.format(
                    spectral_overall[-1]))
                spectral_N = np.sort(np.linalg.eigvals(N))
                print('>>Max eigenvalue of N: {}'.format(spectral_N[-1]))
                print('Min eigenvalue of N: {}'.format(spectral_N[0]))
                spectral_diagN = np.sort(np.linalg.eigvals(diagN))
                print('Max eigenvalue of diagN: {}'.format(
                    spectral_diagN[-1]))
                print('>>Min eigenvalue of diagN: {}'.format(
                    spectral_diagN[0]))
    return examples


class CounterExample():

    def __init__(self, A, z, Q, gap, snr):
        self.A = A
        self.z = z
        self.Q = Q
        self.gap = gap
        self.snr = snr

    def get_noise(self):
        return self.A - self.z.dot(self.z.T)

    def printing(self):
        print('Noise Level: {}'.format(self.snr))
        print('Dual Gap: {}'.format(self.gap))
        print('Noise: ')
        print(self.get_noise())


if __name__ == '__main__':
    examples = search_counter_eg(10, 10, 0, 1, 100)
    for example in examples:
        example.printing()
