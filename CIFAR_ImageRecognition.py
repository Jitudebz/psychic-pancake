# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:10:10 2020

@author: DEBASIS
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle
import sys

#-------------------------------- DATA GATHERING-----------------------------

def load_CIFAR10_batch(filename):
    with open(filename, 'rb') as f:
        if sys.version_info[0] < 3:
            dict = pickle.load(f)
        else:
            dict = pickle.load(f, encoding='latin1')
            x = dict['data']
            y = dict['labels']
            x = x.astype(float)
            y = np.array(y)
    
    return x, y

def load_data():
    xs = []
    ys = []
    for i in range(1, 6):
        filename = 'cifar-10-batches-py/data_batch_' + str(i)
        X, Y = load_CIFAR10_batch(filename)
        xs.append(X)
        ys.append(Y)
        
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del xs, ys
    
    x_test, y_test = load_CIFAR10_batch('cifar-10-batches-py/test_batch')
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck']
    # Normalize Data
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image
    data_dict = {'images_train': x_train,'labels_train': y_train,
                 'images_test': x_test,'labels_test': y_test,'classes': classes}
    
    return data_dict


def reshape_data(data_dict):
    im_tr = np.array(data_dict['images_train'])
    im_tr = np.reshape(im_tr, (-1, 3, 32, 32))
    im_tr = np.transpose(im_tr, (0,2,3,1))
    data_dict['images_train'] = im_tr
    im_te = np.array(data_dict['images_test'])
    im_te = np.reshape(im_te, (-1, 3, 32, 32))
    im_te = np.transpose(im_te, (0,2,3,1))
    data_dict['images_test'] = im_te
    return data_dict

def gen_batch(data, batch_size, num_iter):
    data = np.array(data)
    index = len(data)
    for i in range(num_iter):
        index += batch_size
        if (index + batch_size > len(data)):
            index = 0
            shuffled_indices = np.random.permutation(np.arange(len(data)))
            data = data[shuffled_indices]
            yield data[index:index + batch_size]
            
            
def main():
    data_sets = load_data()
    print(data_sets['images_train'].shape)
    print(data_sets['labels_train'].shape)
    print(data_sets['images_test'].shape)
    print(data_sets['labels_test'].shape)
    if __name__ == '__main__':
        main()









#-----------------------------------DATA MODELLING---------------------------
def inference(images, image_pixels, hidden_units, classes, reg_constant=0):
    # Layer 1
    with tf.variable_scope('Layer1'):
        # Define the variables
        weights = tf.get_variable(name='weights', shape=[image_pixels, hidden_units],
                                  initializer=tf.truncated_normal_initializer(stddev=1.0 / np.sqrt(float(image_pixels))),
                                  regularizer=tf.contrib.layers.l2_regularizer(reg_constant))
        biases = tf.Variable(tf.zeros([hidden_units]), name='biases')
        # Define the layer's output
        hidden = tf.nn.relu(tf.matmul(images, weights) + biases)
    
    # Layer 2
    with tf.variable_scope('Layer2'):
        weights = tf.get_variable('weights', [hidden_units, classes],
                                  initializer=tf.truncated_normal_initializer(stddev=1.0 / np.sqrt(float(hidden_units))),
                                  regularizer=tf.contrib.layers.l2_regularizer(reg_constant))
        biases = tf.Variable(tf.zeros([classes]), name='biases')
        
        logits = tf.matmul(hidden, weights) + biases
        # Define summery-operation for 'logits'-variable
        tf.summary.histogram('logits', logits)
    
    return logits


def loss(logits, labels):
    with tf.name_scope('Loss'):
        # Operation to determine the cross entropy between logits and labels
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                                                      labels=labels, name='cross_entropy'))
        # Operation for the loss function
        loss = cross_entropy + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # Add a scalar summary for the loss
        tf.summary.scalar('loss', loss)
    
    return loss

def training(loss, learning_rate):
    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    # Create a gradient descent optimizer
    # (which also increments the global step counter)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return train_step
    
def evaluation(logits, labels):
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits,1), labels)
        accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('train_accuracy', accuracy)
        
    return accuracy





