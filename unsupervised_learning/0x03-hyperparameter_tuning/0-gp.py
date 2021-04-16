#!usr/env/bin python3 

import  numpy as np

"""
create a gaussian process
"""
class GaussianProcess():
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X,self.X)
    def kernel(self, X1, X2):
        """ calculates the covariance kernel matrix between two matrices """
        """ returns the covariance kernel matrix as a numpy.ndarray of shape (m, n) """
        K = (self.sigma_f**2) * np.exp(np.square(X1 - X2.T) / -(2 * (self.l ** 2)))
        return K



