#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import matplotlib.pyplot as plt
import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
lenet5 = __import__('5-lenet5').lenet5



m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()

X = np.random.uniform(0, 1, (m, h, w, 1))
Y = np.random.randint(0, 10, m)
Y = K.utils.to_categorical(Y, num_classes=10)


model = lenet5(K.Input(shape=(h, w, 1)))
batch_size = 32
epochs = 5
history = model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=False).history
for k, v in sorted(history.items()):
    print(k, v)

