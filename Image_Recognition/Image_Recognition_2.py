# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:16:19 2020

@author: DEBASIS
"""

import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[500], cmap="gray")

#Normalize the train dataset

x_train = tf.keras.utils.normalize(x_train, axis = 1)

x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

#Build input and hidden layers
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

#Build Output `
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

#Compile the model

model.compile(optimizer="adam", 
              loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(x = x_train, y = y_train, epochs=5)

#Evaluate Model

#Evalutate model performance
test_loss, test_acc = model.evaluate(x=x_test, y=y_test)

#Print out the model accuracy
print('\n Test Accuracy:', test_acc)


#Model Predictions
predictions = model.predict([x_test])

plt.imshow(x_test[2345], cmap = "gray")
plt.show()

#Predict

import numpy as np

print(np.argmax(predictions[2345]))


