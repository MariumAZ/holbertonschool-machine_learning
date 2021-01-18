#!/usr/bin/env python3
"""
Creation of confusion matrix
"""
import numpy as  np

def create_confusion_matrix(labels,logits):
    """
    creates a confusion matrix.

    """
    m,classes = labels.shape
    matrix = np.zeros(shape=(classes,classes))
    for i in range(m):
       a = np.where(labels[i,:]==1)
       b = np.where(logits[i,:]==1)
       matrix[a,b] += 1
    return matrix   