#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

class NST():

    """ Neural Style Transfer class """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self,style_image, content_image, alpha=1e4, beta=1):
        self.style_image = style_image
        self.content_image = content_image
        self.alpha = alpha
        self.beta = beta
       
        if (len(self.style_image)) != 3 or (self.style_image.shape[2]!=3):
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (len(self.content_image)) != 3 or (self.style_image.shape[2]!=3):
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (self.alpha < 0):
            raise TypeError("alpha must be a non-negative number")
        if (self.beta < 0):
            raise TypeError("beta must be a non-negative number")   

    @staticmethod
    def scale_image(image):
        if (len(self.image)) != 3 or (self.image.shape[2]!=3):
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * h_new / h)
        else:
            w_new = 512
            h_new = int(h * w_new / w)
        
            
        image = tf.expand_dim(tf.image.resize(images, (h_new,w_new) , method=ResizeMethod.bicubic),axis=0)
        image = image / 255.0
        image = tf.clip_by_value(image,0,1)
        return image
        
        


    




              

