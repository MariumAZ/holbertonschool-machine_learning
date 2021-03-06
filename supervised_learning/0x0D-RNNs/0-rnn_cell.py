#!/usr/bin/env python3`

"""
Performs forward propagation for one time step 

"""
import numpy as np 

class RNNCell:
    def __init__(self, i, h, o):
        self.Wh = np.random.normal(size = (h+i,h))
        self.Wy = np.random.normal(size = (h,o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

        
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=1)    
    def forward(self, h_prev, x_t):
        """
        function taht performs forward propagation for one time step
        """
        #cooncatenate the input with the previous hidden state
        cat = np.concatenate([h_prev, x_t],axis=1)    
        #compute next state
        h_next = np.matmul(cat, self.Wh) + self.bh
        #apply activation  function (non linear  : relu/tanh)
        h_next = np.tanh(h_next)
        #comput output y : if we wanted next sentence (vector of probabilities)
        output = np.matmul(h_next, self.Wy) + self.by
        #apply activation function
        output = self.softmax(output)
        return h_next, output

 
        



        
