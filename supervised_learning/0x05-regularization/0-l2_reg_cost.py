#!/usr/bin/env python3
"""L2 regularization """
import numpy as np

def l2_reg_cost(cost, lambtha, weights, L, m):

    """ function that calculates the cost of a neural
        network with L2 regularization """

    Sum = 0
    for i in range(1, L+1):
        Sum += np.linalg.norm(weights["W"+str(i)])**2
    return cost + (lambtha / (2 * m)) * Sum

    
