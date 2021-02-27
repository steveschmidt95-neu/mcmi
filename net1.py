#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:55:26 2021

@author: stephenschmidt
"""


import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1
import time
import os
from load_data import H5MSI


def weight_variable(name, shape):
    initializer = tf.initializers.GlorotUniform()
    var = tf.Variable(initializer(shape=shape))
    return (var)

def bias_variable(name, shape):
    initializer = tf.zeros_initializer()
    var = tf.Variable(initializer(shape=shape))
    return (var) 

def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=[1], padding='SAME')

def save_conv(sess, test_f1, val_f1):
    directory = 'Net3'
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)
    with open('Results.txt', 'w+') as f:
        f.write('Val Set F1 Score: ' + str(val_f1) + '\n')
        f.write('Test Set F1 Score: ' + str(test_f1) + '\n')
    
    saver = tf.train.Saver()
    saver.save(sess, os.getcwd())
    
class MSInet1(object):
    
    def __init__(self, fc_units = 10, num_classes=2, width1=38, width2=18, width3=16, filters_layer1=6, filters_layer2=6, filters_layer3=6, batch_size=4, fc_keep_prob=.6):
        self.start_time = time.time()
        
        msi_dataset = H5MSI()
        msi_dataset.two_class_data() # remove this for multilable training
        
        self.train_data = msi_dataset.train_files
        self.val_data = msi_dataset.val_files
        
        key = ''
        key_loc = 0
        while 'Labels' not in key and key == '':
            key = list(self.train_data.keys())[key_loc]
            key_loc +=1
                
        self.in_shape = self.train_data[key].shape[1]
        print('Input Shape: ', self.in_shape)
        
        self.fc_unit = fc_units
        self.batch_size = batch_size
        self.fc_keep_prob = fc_keep_prob
        self.num_classes = num_classes
        self.filters_layer1 = filters_layer1
        self.filters_layer2 = filters_layer2
        self.filters_layer3 = filters_layer3
        self.width1 = width1
        self.width2 = width2
        self.width3 = width3
        self.xdev = v1.placeholder(tf.float32, shape=[batch_size, self.in_shape, 1], name = 'xdev')
        self.y_dev = v1.placeholder(tf.int8, shape=[batch_size, self.num_classes],name='y_dev')
        self.fc_keep_prob = v1.placeholder(tf.float32, name = 'fc_keep_prob')
        self.training = v1.placeholder(tf.bool, name='training')
        self.sess = v1.Session()
        self.loss_log = []
        self.accuracy_log = []
        print("Input Shape: ", self.xdev.shape)
        
    
    def buildGraph(self, fc_units = 1000):
        filters_layer1 = self.filters_layer1
        filters_layer2 = self.filters_layer2
        filters_layer3= self.filters_layer3
        in_width = self.in_shape
        
        W_conv1 = weight_variable('W_conv1',[width1, 1, filters_layer1])
        b_conv1 = bias_variable('b_conv1',[filters_layer1])
        
        W_conv2 = weight_variable('W_conv1_2',[width2, filters_layer1, filters_layer2])
        b_conv2 = bias_variable('b_conv1_2',[filters_layer2])
         
        W_conv3 = weight_variable('W_conv1_3',[width3, filters_layer2, filters_layer3])
        b_conv3 = bias_variable('b_conv1_3',[filters_layer3])
        

                
        
        conv1 =   tf.nn.relu(conv1d(self.xdev, W_conv1) + b_conv1) 
        pool1 = tf.nn.max_pool(conv1, 2, 2, padding= "VALID")
        
        conv2 = tf.nn.relu(conv1d(pool1, W_conv2) + b_conv2) 
        pool2 = tf.nn.max_pool(conv2, 2, 2, padding= "VALID")
        
        conv3 =   tf.nn.relu(conv1d(pool2, W_conv3) + b_conv3) 
        pool3 = tf.nn.max_pool(conv3, 2, 2, padding= "VALID")
        
        flat_final_conv = tf.squeeze(pool3)
        
        W_fc1 = weight_variable('W_fc1', [flat_final_conv.shape[1], fc_units])
        b_fc1 = bias_variable('b_fc1',  [fc_units])
        
        W_fc2 = weight_variable('W_fc2', [fc_units, self.num_classes])
        b_fc2 = bias_variable('b_fc2',  [self.num_classes])

        
        fc1 = tf.nn.relu(tf.matmul(flat_final_conv, W_fc1) + b_fc1)
        drop_fc1 = tf.nn.dropout(fc1, self.fc_keep_prob)
        
        fc2 = tf.nn.relu(tf.matmul(drop_fc1, W_fc2) + b_fc2)
        
        self.y_conv = tf.identity(fc2, name='full_op')
        
        
    def train_X_epoch(self, num_epochs = 1, lr = .001,  test_every_epoch = False):
    
        cross_entropy = v1.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_dev, logits=self.y_conv, dim = -1)
        cross_entropy_sum = tf.reduce_sum(cross_entropy)
        train_step = v1.train.AdamOptimizer(learning_rate = lr).minimize(cross_entropy_sum)
        cost = cross_entropy_sum

batch_size = 3
fc_keep_prob = 1
filters_layer1 = 1
filters_layer2 = 1
filters_layer3 = 1
width1 = 38
width2 = 18
width3 = 16
num_classes= 2
num_epochs = 1
fc_units = 10


net1 = MSInet1(fc_units=fc_units, num_classes=num_classes, width1=width1, width2=width2, width3=width3,filters_layer1=filters_layer1, filters_layer2=filters_layer2, filters_layer3=filters_layer3, batch_size=batch_size, fc_keep_prob=fc_keep_prob)  
net1.buildGraph()
net1.train_X_epoch()
