#!/usr/bin/env python3

import tensorflow.keras as K
def preprocess_data(X,Y):
  X = K.applications.densenet.preprocess_input(X)
  Y = K.utils.to_categorical(Y, 10)
  return X,Y

input1 = K.layers.Input(shape=(32,32,3))
input = K.layers.Lambda(lambda image: tf.image.resize(image,(224, 224)))(input1)
dense = K.applications.DenseNet121(include_top=False,weights='imagenet',input_tensor=input)
output = dense.output
#x = k.layers.Lambda(lambda i:K.resize_images(i,224,224,data_format="channels_last"))(output)
x = K.layers.GlobalAveragePooling2D()(output)
x = K.layers.Dense(512,activation='relu')(x)
x = K.layers.Dense(10,activation='softmax')(x)
Model = K.models.Model(inputs=input1,outputs=x)
Model.compile(loss='categorical_crossentropy',optimizer=K.optimizers.RMSprop(lr=2e-5),metrics=['accuracy'])
for layer in Model.layers[:-3]:
  layer.trainable = False
for layer in Model.layers[-3:]:
  layer.trainable = True  
check = K.callbacks.ModelCheckpoint(filepath="cifar10.h5",
                                        monitor="val_acc",
                                        mode="max",
                                        save_best_only=True)  
Model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=1,batch_size=128, verbose=1,callbacks=[check]) 


