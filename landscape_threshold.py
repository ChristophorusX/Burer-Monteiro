# This piece of program generates the adjacent matrix of underlying ER graph,
# then with the adjacency matrix as observation we form the Buer-Monteiro
# problem in trigonometric representation. For each size of the system n and
# probability around log n/n, with test on 10 random cases and plot the
# success rate.

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def adj_matrix_generator(n, p):
    G = nx.erdos_renyi_graph(n, p)
    return nx.adjacency_matrix(G)


def local_method(A):
    Q, theta = gradient_descent(A, 0.005)
    theta = np.mod(theta, 2 * np.pi)
    diff = np.abs(theta - theta[0])
    inf_norm_diff = np.max(diff)
    if inf_norm_diff < 10**(-4):
        return 1
    else:
        return 0


def gradient_descent(A, step_size, max_iteration=1000):
    dim, _ = A.shape
    target_val = np.trace(A.dot(np.ones((dim, dim))))
    init = np.random.uniform(-np.pi, np.pi, size=dim)
    theta = init
    Q = np.empty((dim, 2))
    for i in range(max_iteration):
        x = np.cos(theta, out=None)
        y = np.sin(theta, out=None)
        Q = (np.vstack((x, y))).transpose()
        grad = 2 * (A.dot(x) * y - A.dot(y) * x)
        theta -= step_size * grad
        obj_val = -np.trace(A.dot(Q.dot(Q.T)))
        if np.abs(target_val - obj_val) < 0.1:
            break
    return Q, theta


def working_loop(n_min, n_max, n_step, p_min, p_max, p_step, n_sample):
    n_row = int((n_max - n_min) / n_step)
    n_col = int((p_max - p_min) / p_step)
    result = np.zeros((n_row, n_col))
    np.save("result-array-large", np.rot90(result))
    for row in range(0, n_row):
        n = n_min + n_step * row
        print("Working on n = {}...\n".format(n))
        for col in range(0, n_col):
            p = p_min + p_step * col
            prob = min(p * np.log(n) / n, 1)
            print("Working on p = {}, prob = {}".format(p, prob))
            n_success = 0
            for sample in range(n_sample):
                A = adj_matrix_generator(n, prob)
                result_indicator = local_method(A)
                n_success += result_indicator
                print("|", end="", flush=True)
            success_rate = n_success / n_sample
            result[row, col] = success_rate
            print("-> Success rate: {}".format(success_rate))
            print('')
        print(result[row])
        print('')
        np.save("result-array-large", np.rot90(result))
    return result


if __name__ == '__main__':
    # A = adj_matrix_generator(100, 0.8)
    # result_indicator = local_method(A)
    # print(result_indicator)

    # result = working_loop(100, 1050, 50, 1, 2.1, 0.1, 50)
    # print(result)
    # np.save("result-array-large", np.rot90(result))

    # result = np.load("result-array-new.npy")
    # plt.matshow(result, fignum=None)
    # plt.savefig("success-rate-plot-new.png", dpi=200)

    result1 = np.load("result-array-large.npy")
    # print(result1)
    result2 = np.load("result-array-large-extra.npy")
    # print(result2)
    result3 = np.load("result-array-large-extra2.npy")
    # print(result3)
    result = np.vstack((result3, result2, result1))
    # print(result)
    a, b = result.shape
    zero_block = np.zeros((10, b))
    one_block = np.ones((14, b))
    final = np.vstack((one_block, result, zero_block))
    print(final)
    np.save("phase-transition-large", final)
