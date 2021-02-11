#!/usr/bin/env python3

import tensorflow.keras as k
def preprocess_data(X,Y):
  X = k.applications.densenet.preprocess_input(X)
  Y = k.utils.to_categorical(Y, 10)
  return X,Y

input1 = k.layers.Input(shape=(32,32,3))
input = k.layers.Lambda(lambda image: tf.image.resize(image,(224, 224)))(input1)
dense = k.applications.DenseNet121(include_top=False,weights='imagenet',input_tensor=input)
output = dense.output
#x = k.layers.Lambda(lambda i:K.resize_images(i,224,224,data_format="channels_last"))(output)
x = k.layers.GlobalAveragePooling2D()(output)
x = k.layers.Dense(512,activation='relu')(x)
x = k.layers.Dense(10,activation='softmax')(x)
Model = k.models.Model(inputs=input1,outputs=x)
Model.summary()
Model.compile(loss='categorical_crossentropy',optimizer=k.optimizers.RMSprop(lr=2e-5),metrics=['accuracy'])
for layer in Model.layers[:-3]:
  layer.trainable = False
for layer in Model.layers[-3:]:
  layer.trainable = True  
Model.fit(x_train,y_train,epochs=5,batch_size=32, verbose=1) 

