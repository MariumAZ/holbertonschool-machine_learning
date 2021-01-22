#!/usr/bin/env python3
""" l2 regularization function """

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ function that creates a layer of a neural network
        using dropout"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    linear_model = tf.layers.Dense(units=n, activation=activation,
                                   kernel_initializer=init)(prev)
    dropout_layer = tf.layers.Dropout(keep_prob)(linear_model)
    return dropout_layer
