from part1 import *
import numpy as np
import math



Xtest = np.mat([[0., 1.],
                [1., 0.],
                [2., 2.],
                [1., 1.]])

w = np.mat(' 0.5; -0.6')
b = 0.2


n = Xtest.shape[0]
Y = np.zeros(n)  # initialize as all zeros
P = np.mat(np.zeros((n, 1)))
for i, x in enumerate(Xtest):
    x = x.T  # convert to column vector
    #########################################
    ## INSERT YOUR CODE HERE
    z = compute_z(x, w, b)
    if z > 0:
        y = 1
    elif z < 0:
        y = 0

print(Y.shape)
print(P.shape)