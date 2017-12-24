import numpy as np


def uniform_noise(n, level):
    ran = np.random.uniform(-level, level, (n, n))
    ran = ran - np.diag(np.diag(ran))
    ran = np.triu(ran) + (np.triu(ran)).T
    return ran


if __name__ == '__main__':
    A = uniform_noise(10, 1)
    print(A)
