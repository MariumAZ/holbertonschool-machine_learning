#!/usr/bin/env python3
import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell

def rnn(rnn_cell, X, h_0):
    H = [h_0]
    Y = []
    for x_t in X:
        h_next, y = rnn_cell.forward(H[-1],x_t)
        H.append(h_next)
        Y.append(y)
    return np.array(H), np.array(Y)      

