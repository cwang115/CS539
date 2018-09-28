import numpy as np
from linear_regression import *

x = np.mat('1.;2.;3')

result = compute_Phi(x, 2)
result = np.array(result)
print(result)

y = np.mat(' 0.5; 0.2; 1')
yhat = np.mat(' 0.56; 0.3; 0.95')
Phi = np.mat(' 2., 0.8; 5., 0.9; 6., 3.')

dL_dw = compute_dL_dw(y, yhat, Phi)
print(dL_dw)

dL_dw_true =np.mat('0.10667; -0.004')
print(dL_dw_true )