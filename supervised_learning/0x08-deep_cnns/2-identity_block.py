#!/usr/bin/env python3
import tensorflow.keras as K

def identity_block(A_prev, filters):
    init = K.initializers.he_normal()
    F11, F3, F12 = filters
    layer = K.layers.Conv2D(F11,(1,1),strides=1,padding='same',kernel_inizializer=init)(A_prev)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Activation(activation='relu')(layer)
    #second layer 
    layer = K.layers.Conv2D(F3,(3,3),strides=1,padding='same',kernel_inizializer=init)(layer)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Activation(activation='relu')(layer)
    #third layer
    layer = K.layers.Conv2D(F12,(1,1),strides=1,padding='same',kernel_inizializer=init)(layer)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Add()([layer, A_prev])
    layer = K.layers.Activation(activation='relu')(layer)
    return layer





    
