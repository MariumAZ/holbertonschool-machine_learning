#!/usr/bin/env python3
import tensorflow.keras as K

def projection_block(A_prev, filters, s=2):
    init = K.initializers.he_normal()
    F11, F3, F12 = filters
    layer = K.layers.Conv2D(F11,(1,1),strides=s,padding='same',kernel_initializer=init)(A_prev)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Activation(activation='relu')(layer)
    #second layer 
    layer = K.layers.Conv2D(F3,(3,3),padding='same',kernel_initializer=init)(layer)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Activation(activation='relu')(layer)
    #third layer
    layer = K.layers.Conv2D(F12,(1,1),padding='same',kernel_initializer=init)(layer)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    #Adding layer
    layer_2 = K.layers.Conv2D(F12,(1,1),strides=s,padding='same',kernel_initializer=init)(A_prev)
    layer_2 = K.layers.BatchNormalization(axis=3)(layer_2)
    layer = K.layers.Add()([layer, layer_2])
    layer = K.layers.Activation(activation='relu')(layer)
    return layer