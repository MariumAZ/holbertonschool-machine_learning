#!/usr/bin/env python3

import numpy as np
def l2_reg_cost(cost, lambtha, weights, L, m):
    """ function that calculates the cost of a neural
        network with L2 regularization """
        weight_sum = 0
        for i in range(1, L+1):
            #numpy.linalg.norm is used to calculate the norm of a vector or a matrix
            weight_sum += np.linalg.norm(weights.get(["W"+str(i)]))
        return weight_sum + (lambtha/ (2 * m)) * weight_sum

