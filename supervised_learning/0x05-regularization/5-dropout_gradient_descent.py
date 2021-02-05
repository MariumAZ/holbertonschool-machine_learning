#!/usr/bin/env python3
""" Gradient Descent Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    :param X: numpy.ndarray of shape (nx, m) containing
        the input data for the network
        nx is the number of input features
        m is the number of data pointsnumpy.ndarray of shape (nx, m)
        containing the input data for the network
    :param weights: dictionary of the weights and biases of the neural network
    :param L:
    :param keep_prob: probability that a node will be kept
    :return: dictionary containing the outputs of each layer and the dropout
        mask used on each layer (see example for format)
    """
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        dw = (1 / m) * np.matmul(dz, cache["A" + str(i-1)].T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dA = 1 - cache["A" + str(i-1)] * cache["A" + str(i-1)]
        if i>1:
            dz = np.matmul(weights["W" + str(i)].T, dz) * dA * cache["D"+str(i-1)] / keep_prob
        weights["W" + str(i)] = weights["W" + str(i)] - alpha * dw
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db