import numpy as np

result = np.load("result-array-new.npy")
result.astype('float32').tofile('result-array-new.dat')
