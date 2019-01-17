import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE
#print(Xtrain, Ytrain)
learning_rate = 0.05
epochs = 1000
print("Training results for linear regression")
para_info = "Learning rate = {}, Number of epochs = {}".format(learning_rate,epochs)
print(para_info)
w = train(Xtrain, Ytrain, learning_rate, epochs)
train_yhat = Xtrain.dot(w)
test_yhat = Xtest.dot(w)
train_loss = compute_L(train_yhat,Ytrain)
test_loss = compute_L(test_yhat, Ytest)
print("Final training loss:", train_loss.item(0))
print("Test loss:", test_loss.item(0))
#########################################

