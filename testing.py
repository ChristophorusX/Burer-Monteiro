import burer_monteiro as bm
import numpy as np
import spectral_gap_analysis as analysis
import aux
import noise_generator as gen
import sdp

if __name__ == '__main__':
    n = 10
    k = 2
    level = 5

    while True:
        z = np.ones(n).reshape((-1, 1))
        ground_truth = z.dot(z.T)
        # N = analysis._gen_row_mat(n, level)
        N = gen.uniform_noise(n, 10) + 10
        N = N - np.diag(np.diag(N))
        gap = aux.laplacian_eigs(N, z)[1]
        print('Spectral gap of perturbation: {}'.format(gap))
        A = ground_truth + N
        relaxation_val, _, _ = sdp.sdp_relaxation(A, z)
        relaxation_val_minus, _, _ = sdp.sdp_relaxation(-A, z)
        relaxation_val_N, _, _ = sdp.sdp_relaxation(N, z)
        relaxation_val_minus_N, _, _ = sdp.sdp_relaxation(-N, z)
        print('sdp(A) = {}'.format(relaxation_val))
        print('sdp(-A) = {}'.format(relaxation_val_minus))
        print('sdp(N) = {}'.format(relaxation_val_N))
        print('sdp(-N) = {}'.format(relaxation_val_minus_N))
        print('Smallest eigenvalue of A = {}'.format(np.linalg.eigvals(A)[0]))
        Q = bm._vector_to_matrix(bm.trust_region(A, k, plotting=False, printing=False).x, k)
        # print(np.linalg.norm(Q.dot(Q.T) - ground_truth))
        Q_dot = Q
        Q_dot[:, 0] = Q[:, 1]
        Q_dot[:, 1] = Q[:, 0]
        left = (np.diag(np.diag((A.dot(Q)).dot(Q.T))) - A).dot(Q_dot) - (np.diag(np.diag((A.dot(Q_dot)).dot(Q.T))) - A).dot(Q)
        inner = np.trace((left.reshape((k, n))).dot(Q_dot))
        # print(inner)
        l = Q_dot - np.diag(np.diag(Q_dot.dot(Q.T))).dot(Q)
        inner2 = np.trace((l.reshape((k, n))).dot(Q_dot))
        # print(inner - inner2)
        mat = (np.diag(np.diag(A.dot(ground_truth)))) - A * ground_truth
        # print(np.sort(np.linalg.eigvals(mat)))
        Q = Q / np.max(np.linalg.norm(Q, 2, axis=1))
        AQ = A.dot(Q)
        u = np.random.random_sample((n, k))
        u = u / np.linalg.norm(u, 2)
        D = np.diag(np.linalg.norm(u, 2, axis=1))
        u = D.dot(u)
        print('IF NORMALIZED: {}'.format(np.linalg.norm(u, 2)))
        print('IF NORMALIZED: {}'.format(np.linalg.norm(Q, 2, axis=1)))
        prod = np.abs(np.sum(AQ * u))
        ma = np.max(np.linalg.norm(AQ, 2, axis=1))
        print('Inner product = {}'.format(prod))
        print('Max norm = {}'.format(ma))
