# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:59:19 2020

@author: DEBASIS
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
from datetime import datetime
import data_gathering
import two_layer_fc
import sys
import numpy as np

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

# Basic model parameters as external flags.
flags = tf.app.flags

flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_string('train_dir', 'tf_logs', 'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')


FLAGS = tf.app.flags.FLAGS
FLAGS(sys.argv)
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{} = {}'.format(attr.upper(), value))
print()


IMAGE_PIXELS = 3072
CLASSES = 10

beginTime = time.time()

data_sets = data_gathering.load_data()

images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], 
                                    name='images')

labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')

tf.reset_default_graph()

hidden_units = FLAGS.hidden1
image_pixels = IMAGE_PIXELS
images = images_placeholder
classes = CLASSES
reg_constant=FLAGS.reg_constant



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



logits = two_layer_fc.inference(images_placeholder, IMAGE_PIXELS,
                                FLAGS.hidden1, CLASSES,
                                reg_constant=FLAGS.reg_constant)

global_step = tf.Variable(0, name="global_step", trainable=False)

accuracy = two_layer_fc.evaluation(logits, labels_placeholder)

saver = tf.train.Saver()


with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring variables from checkpoint')
        saver.restore(sess, ckpt.model_checkpoint_path)
        current_step = tf.train.global_step(sess, global_step)
        print('Current step: {}'.format(current_step))
    print('Test accuracy {:g}'.format(accuracy.eval(feed_dict={images_placeholder: data_sets['images_test'],
          labels_placeholder: data_sets['labels_test']})))
  
  
endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))