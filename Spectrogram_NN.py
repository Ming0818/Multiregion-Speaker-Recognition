import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from create_speak_reco_ds import dataset_train
from create_speak_reco_ds import dataset_test
from tf_utils import random_mini_batches
from tensorflow import keras

X_train_orig, Y_train_orig = dataset_train()
X_test_orig, Y_test_orig = dataset_test()

sess = tf.Session()
X_train = X_train_orig/255.0
X_test = X_test_orig/255.0

print np.shape(X_train)
with sess.as_default():
    Y_train = tf.one_hot(Y_train_orig, 15).eval()
    Y_test = tf.one_hot(Y_test_orig, 15).eval()
print np.shape(Y_train)


model = tf.keras.Sequential([keras.layers.Flatten(),
                             keras.layers.Dense(512, activation=tf.nn.leaky_relu),
                             keras.layers.Dense(400, activation=tf.nn.relu),
                             keras.layers.Dense(15, activation=tf.nn.softmax)])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=100)

model.evaluate(X_test, Y_test)

