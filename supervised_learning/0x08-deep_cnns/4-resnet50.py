#!/usr/bin/env python3
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block
def resnet50():
    init = K.initializers.he_normal()
    Input = K.Input(shape=(224,224,3))
    layer = K.layers.Conv2D(64,(7,7),strides=2,padding='same',kernel_initializer=init)(Input)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Activation(activation='relu')(layer)
    layer = K.layers.MaxPooling2D((3,3),strides=2,padding='same')(layer)
    
    layer = projection_block(layer, [64, 64, 256], 1)

    for i in range(2):
        layer = identity_block(layer, [64, 64, 256])

    layer = projection_block(layer, [128, 128, 512])

    for i in range(3):
        layer = identity_block(layer, [128, 128, 512])

    layer = projection_block(layer, [256, 256, 1024])

    for i in range(5):
        layer = identity_block(layer, [256, 256, 1024])

    layer = projection_block(layer, [512, 512, 2048])

    for i in range(2):
        layer = identity_block(layer, [512, 512, 2048])
    
    out = K.layers.Dense(1000,
                         activation='softmax',
                         kernel_regularizer=K.regularizers.l2())(layer)

    model = K.models.Model(inputs=Input, outputs=out)

    return model    






