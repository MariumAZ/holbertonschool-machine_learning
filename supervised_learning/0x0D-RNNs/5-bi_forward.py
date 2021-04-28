

#!/usr/bin/env python3
"""
new class
"""
import numpy as np
class BidirectionalCell:
    """BidirectionalCell class"""
    def __init__(self, i, h, o):
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))
    def forward(self, h_prev, x_t):
        stacked = np.hstack((h_prev, x_t))
        h_next = np.tanh(stacked @ self.Whf + self.bhf)
        return h_next    
    def backward(self, h_next, x_t):
        stacked = np.hstack((h_next, x_t))
        h_prev = np.tanh(stacked @ self.Whb + self.bhb)
        return h_prev