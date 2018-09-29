import numpy as np
from linear_regression import *

Xtrain = np.mat([[0., 1.],
                 [1., 0.],
                 [0., 0.],
                 [1., 1.]])
Ytrain = np.mat('0.1; 0.2; 0; 0.3')

w = train(Xtrain, Ytrain, alpha=1., n_epoch=20)

print(Xtrain.dot(w))

print(Ytrain)

