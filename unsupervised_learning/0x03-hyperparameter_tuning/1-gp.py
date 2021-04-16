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
    def predict(self, X_s):
        """ that predicts the mean and standard deviation of points in a Gaussian process: """
        """ X_s is a numpy.ndarray of shape (s, 1) containing all of the points """
        """Returns: mu, sigma"""
        K_s = self.kernel(self.X, X_s)
        K_inv = np.linalg.inv(self.K)
        #mu_s = np.matmul(np.matmul(K_s.T, K_inv), self.Y).reshape(-1)
        mu_s =np.squeeze(K_s.T.dot(K_inv).dot(self.Y))
        #sig_s = self.sigma_f**2 - np.sum(np.matmul(K_s.T, K_inv).T * K_s, axis=0)
        K_ss = self.kernel(X_s, X_s)
        #print(np.matmul(np.matmul(K_s.T, K_inv), K_s))
        #print(k_ss.shape)
        #sig_s = k_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s)
        sig_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sig_s = np.diag(sig_s)

        return mu_s, sig_s
        
        

