#!/usr/bin/env python3
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block

def inception_network():
    Input = K.Input(shape=(224,224,3))
    layer_1 = K.layers.Conv2D(64,(7,7),strides=2,activation='relu',padding='same')(Input)
    layer_1 = K.layers.MaxPooling2D((3,3),strides=2,padding='same')(layer_1)
    layer_1 = K.layers.Conv2D(64,(1,1),strides=1,activation='relu')(layer_1)
    layer_1 = K.layers.Conv2D(192,(3,3),strides=1,activation='relu',padding='same')(layer_1)
    layer_1 = K.layers.MaxPooling2D((3,3),strides=2,padding='same')(layer_1)
    filters = [64, 96, 128, 16, 32, 32]
    layer_1 = inception_block(layer_1,filters)
    filters = [128, 128, 192, 32, 96, 64]
    layer_1 = inception_block(layer_1,filters)
    layer_1 = K.layers.MaxPooling2D((3,3),strides=2,padding='same')(layer_1)
    filters = [192,96,208,16,48,64]
    layer_1 = inception_block(layer_1,filters)
    filters = [160,112,224,24,64,64]
    layer_1 = inception_block(layer_1,filters)
    filters = [128,128,256,24,64,64]
    layer_1 = inception_block(layer_1,filters)
    filters = [112,144,288,32,64,64]
    layer_1 = inception_block(layer_1,filters)
    filters = [256,160,320,32,128,128]
    layer_1 = inception_block(layer_1,filters)
    layer_1 = K.layers.MaxPooling2D((3,3),strides=2,padding='same')(layer_1)
    filters = [256,160,320,32,128,128]
    layer_1 = inception_block(layer_1,filters)
    filters = [384,192,384,48,128,128]
    layer_1 = inception_block(layer_1,filters)
    layer_1 = K.layers.AveragePooling2D((7,7),strides=1)(layer_1)
    layer_1 = K.layers.Dropout(0.4)(layer_1)
    #layer_1 = K.layers.Flatten()(layer_1)
    layer_1 = K.layers.Dense(1000,activation='softmax')(layer_1)
    model = K.models.Model(inputs=Input, outputs=layer_1)
    return model
   



