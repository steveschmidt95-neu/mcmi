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
    
    def __init__(self, fc_units = 10, num_classes=2, width1=38, width2=18, width3=16, filters_layer1=6, filters_layer2=6, filters_layer3=6, batch_size=4):
        self.start_time = time.time()
        
        msi_dataset = H5MSI()
        
        if num_classes == 2:
            msi_dataset.two_class_data() # remove this for multilable training
        msi_dataset.flatten_data()
        
        self.flat_train = msi_dataset.flat_train
        self.flat_train_labels = msi_dataset.flat_train_labels
        self.flat_val = msi_dataset.flat_val
        self.flat_val_labels = msi_dataset.flat_val_labels
        self.in_shape = self.flat_train.shape[1]
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
        self.y_dev = v1.placeholder(tf.float32, shape=[batch_size, self.num_classes],name='y_dev')
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

        
        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        conv1 =   tf.nn.relu(conv1d(self.xdev, W_conv1) + b_conv1) 
        pool1 = tf.nn.max_pool(conv1, 2, 2, padding= "VALID")

        conv2 = tf.nn.relu(conv1d(pool1, W_conv2) + b_conv2) 
        pool2 = tf.nn.max_pool(conv2, 2, 2, padding= "VALID")
        
        conv3 =   tf.nn.relu(conv1d(pool2, W_conv3) + b_conv3) 
        pool3 = tf.nn.max_pool(conv3, 2, 2, padding="VALID")
        flat_final_conv = v1.layers.flatten(pool3)
        
        W_fc1 = weight_variable('W_fc1', [flat_final_conv.shape[1], fc_units])
        b_fc1 = bias_variable('b_fc1',  [fc_units])
        
        W_fc2 = weight_variable('W_fc2', [fc_units, self.num_classes])
        b_fc2 = bias_variable('b_fc2',  [self.num_classes])
        
        fc1 = tf.nn.relu(tf.matmul(flat_final_conv, W_fc1) + b_fc1)
        #drop_fc1 = tf.nn.dropout(fc1, self.fc_keep_prob, training=self.training)
        drop_fc1 = v1.layers.dropout(fc1, self.fc_keep_prob, training=self.training)

        fc2 = tf.nn.relu(tf.matmul(drop_fc1, W_fc2) + b_fc2)
        
        self.y_conv = tf.identity(fc2, name='full_op')
        
        
    def train_X_epoch(self, num_epochs = 1, lr = .001, keep_prob=.8,  test_every_epoch = False, x_epoch=5):
    
        cross_entropy = v1.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_dev, logits=self.y_conv)
        cross_entropy_sum = tf.reduce_sum(cross_entropy)
        train_step = v1.train.AdamOptimizer(learning_rate = lr).minimize(cross_entropy_sum)
        cost = cross_entropy_sum
        
        print('---------------------Starting Training------------------')
        self.sess.run(v1.global_variables_initializer())
        
        for epoch in range(num_epochs):
            epoch_loss_sum = 0
            batch_idx = 0
            
            
            # Clip Training for testing
            self.flat_train = self.flat_train[0:100, :]
            
            
            while (batch_idx < self.flat_train.shape[0] and batch_idx+self.batch_size < self.flat_train.shape[0]):
                print('BatchIDX: ', batch_idx, ' / ', self.flat_train.shape[0])
                
                train_batch = self.flat_train[batch_idx:batch_idx+self.batch_size,:]
                train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
                print(train_batch.shape)
                train_labels = self.flat_train_labels[batch_idx:batch_idx+self.batch_size]
                batch_idx += self.batch_size
                
                feed_dict={self.xdev: train_batch, self.y_dev: train_labels, self.fc_keep_prob: keep_prob, self.training: True }
                [_, cross_entropy_py] = self.sess.run([train_step, cost], feed_dict=feed_dict)
                self.loss_log.append(cross_entropy_py)
                epoch_loss_sum += cross_entropy_py
                
            sci_loss = '%e' % epoch_loss_sum
            print('Epoch ' + str(epoch+1) + ' Loss: ' + str(sci_loss))
            
            if test_every_epoch and (epoch%x_epoch == 0) and epoch !=0:
                
                batch_idx = 0
                true_pos = 0
                true_neg = 0
                false_pos = 0
                false_neg = 0
                
                while (batch_idx < self.flat_val.shape[0] and batch_idx+self.batch_size < self.flat_val.shape[0]):
                    val_batch = self.flat_val[batch_idx:batch_idx+self.batch_size,:]
                    val_batch = np.reshape(val_batch, (val_batch.shape[0], val_batch.shape[1], 1))
                    val_labels = self.flat_val_labels[batch_idx:batch_idx+self.batch_size]  
                    batch_idx += self.batch_size
                    
                    #feed_dict={self.xdev: val_batch, self.fc_keep_prob: 1, self.training: False}
    
                    feed_dict={self.xdev: val_batch, self.fc_keep_prob: keep_prob, self.training: False }

                    classification = self.sess.run(v1.nn.softmax(self.y_conv), feed_dict)
                    
                    print(classification.shape)
                    print(classification)
                    
                    assert False
                

batch_size = 10
fc_keep_prob = 1
filters_layer1 = 4
filters_layer2 = 8
filters_layer3 = 16
width1 = 38
width2 = 18
width3 = 16
num_classes= 4
num_epochs = 50
fc_units = 10
keep_prob=.9
test_every_epoch = True
x_epoch = 1
lr = .00001

# for two class labels, 0 is healthy 1 is cancerous
# For multi Class,
#         self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
# Shifted back 1
#        self.diagnosis_dict = {'high': 0, 'CA': 1, 'low': 2, 'healthy': 3}




net1 = MSInet1(fc_units=fc_units, num_classes=num_classes, width1=width1, width2=width2, width3=width3,filters_layer1=filters_layer1, filters_layer2=filters_layer2, filters_layer3=filters_layer3, batch_size=batch_size)  
net1.buildGraph()
net1.train_X_epoch(lr=lr, keep_prob=keep_prob, num_epochs=num_epochs, test_every_epoch=test_every_epoch, x_epoch=x_epoch)
