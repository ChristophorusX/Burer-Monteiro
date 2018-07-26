# This piece of program generates the adjacent matrix of underlying ER graph,
# then with the adjacency matrix as observation we form the Buer-Monteiro
# problem in trigonometric representation. For each size of the system n and
# probability around log n/n, with test on 10 random cases and plot the
# success rate.

import numpy as np
import networkx as nx
import trigonometric as trig
import matplotlib.pyplot as plt


def adj_matrix_generator(n, p):
    G = nx.erdos_renyi_graph(n, p)
    return nx.adjacency_matrix(G)


def local_method(A):
    Q, theta = trig.trig_bfgs(A, None, init=None)
    theta = np.mod(theta, 2 * np.pi)
    diff = np.abs(theta - theta[0])
    inf_norm_diff = np.max(diff)
    if inf_norm_diff < 10**(-4):
        return 1
    else:
        return 0


def working_loop(n_min, n_max, n_step, p_min, p_max, p_step, n_sample):
    n_row = int((n_max - n_min) / n_step)
    n_col = int((p_max - p_min) / p_step)
    result = np.empty([n_row, n_col])
    for row in range(0, n_row):
        n = n_min + n_step * row
        print("Working on n = {}...\n".format(n))
        for col in range(0, n_col):
            p = p_min + p_step * col
            # prob = min(2**p * np.log(n) / n, 1)
            print("Working on p = {}, prob = {}".format(p, p))
            n_success = 0
            for sample in range(n_sample):
                A = adj_matrix_generator(n, p)
                result_indicator = local_method(A)
                n_success += result_indicator
                print("|", end='', flush=True)
            success_rate = n_success / n_sample
            result[row, col] = success_rate
            print("-> Success rate: {}".format(success_rate))
            print('')
        print(result[row])
        print('')
    return result


if __name__ == '__main__':
    # A = adj_matrix_generator(100, 0.8)
    # result_indicator = local_method(A)
    # print(result_indicator)

    result = working_loop(10, 110, 10, 0, 1.1, 0.1, 50)
    print(result)
    result = np.rot90(result)
    np.save("result-array", result)
    plt.matshow(result, fignum=None)
    plt.savefig("success-rate-plot.png", dpi=200)
