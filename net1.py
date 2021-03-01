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
        
    def get_next_batch(self):
        train_with_labels = np.zeros((self.flat_train.shape[0], self.flat_train.shape[1] + self.num_classes))
        train_with_labels[:, 0:self.flat_train.shape[1]] = self.flat_train
        train_with_labels[:, self.flat_train.shape[1]:self.flat_train.shape[1]+self.num_classes]= self.flat_train_labels[:, :]
        np.random.shuffle(train_with_labels)
        
        self.flat_train = train_with_labels[:, 0:self.flat_train.shape[1]]
        self.flat_train_labels = train_with_labels[:, ((self.num_classes)*-1):]
        
        
        train_batch = np.zeros((self.batch_size, self.flat_train.shape[1]))
        train_labels = np.zeros((self.batch_size, self.num_classes))
        
        batch_div = self.batch_size//self.num_classes
        
        for class_idx in range(0,self.num_classes):
    
            class_locs = np.where(self.flat_train_labels[:, class_idx]==1)[0:batch_div]
            class_locs = class_locs[0][0:batch_div]
            class_values = self.flat_train[class_locs[0:batch_div], :]
            class_labels = self.flat_train_labels[class_locs, :]
            
            train_batch[((self.batch_size//self.num_classes)*class_idx):(batch_div)*class_idx +batch_div, :] = class_values
            train_labels[((self.batch_size//self.num_classes)*class_idx):(batch_div)*class_idx +batch_div, :] = class_labels
            
        train_with_labels = np.zeros((train_batch.shape[0], train_batch.shape[1] + self.num_classes))
        train_with_labels[:, 0:train_batch.shape[1]] = train_batch
        train_with_labels[:, train_batch.shape[1]:train_batch.shape[1]+self.num_classes]= train_labels
        np.random.shuffle(train_with_labels)
        
        train_batch = train_with_labels[:, 0:train_batch.shape[1]]
        train_labels = train_with_labels[:, ((self.num_classes)*-1):]

            
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        return(train_batch, train_labels)
 
        
        
    def train_X_epoch(self, num_epochs = 1, lr = .001, keep_prob=.8,  test_every_epoch = False, x_epoch=5, shuffle=False):
    
        cross_entropy = v1.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_dev, logits=self.y_conv)
        cross_entropy_sum = tf.reduce_sum(cross_entropy)
        train_step = v1.train.AdamOptimizer(learning_rate = lr).minimize(cross_entropy_sum)
        cost = cross_entropy_sum
        
        print('---------------------Starting Training------------------')
        self.sess.run(v1.global_variables_initializer())
        
        for epoch in range(num_epochs):
            epoch_loss_sum = 0

            train_batch, train_labels = self.get_next_batch()      


            feed_dict={self.xdev: train_batch, self.y_dev: train_labels, self.fc_keep_prob: keep_prob, self.training: True }
            [_, cross_entropy_py] = self.sess.run([train_step, cost], feed_dict=feed_dict)
            epoch_loss_sum += cross_entropy_py
                
            sci_loss = '%e' % epoch_loss_sum
            print('')
            print('Epoch ' + str(epoch+1) + ' Loss: ' + str(sci_loss))
            
            if test_every_epoch and ((epoch+1)%x_epoch == 0) and epoch !=0:
                
                batch_idx = 0
                label_predictions = np.zeros_like(self.flat_val_labels)

                while (batch_idx < self.flat_val.shape[0] and batch_idx+self.batch_size < self.flat_val.shape[0]):

                    val_batch = self.flat_val[batch_idx:batch_idx+self.batch_size,:]
                    val_batch = np.reshape(val_batch, (val_batch.shape[0], val_batch.shape[1], 1))
                    
                    feed_dict={self.xdev: val_batch, self.fc_keep_prob: keep_prob, self.training: False }

                    classification = self.sess.run(v1.nn.softmax(self.y_conv), feed_dict)
                    label_predictions[batch_idx:batch_idx+self.batch_size, :] = classification
                    batch_idx += self.batch_size
                
                remaining = (self.flat_val.shape[0] - batch_idx)
                if (remaining > 0):
                    val_batch = np.zeros((self.batch_size, self.flat_val.shape[1]))
                    val_batch[0:remaining, :] = self.flat_val[batch_idx:batch_idx+remaining, :]
                val_batch = np.reshape(val_batch, (val_batch.shape[0], val_batch.shape[1], 1))
                feed_dict={self.xdev: val_batch, self.fc_keep_prob: keep_prob, self.training: False}
                classification = self.sess.run(v1.nn.softmax(self.y_conv), feed_dict)
                label_predictions[batch_idx:batch_idx+remaining, :] = classification[0:remaining, :]
                
                self.metric_validation(label_predictions)


                
    def metric_validation(self, predictions):
        one_hot_predictions= np.zeros_like(predictions)
        one_hot_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1
        
        mult = self.flat_val_labels * one_hot_predictions
        
        if self.num_classes == 4:
        
            class1_total = np.sum(self.flat_val_labels[:,0])
            class2_total = np.sum(self.flat_val_labels[:,1])
            class3_total = np.sum(self.flat_val_labels[:,2])
            class4_total = np.sum(self.flat_val_labels[:,3])
            
            class1_correct = np.sum(mult[:,0])
            class2_correct = np.sum(mult[:,1])
            class3_correct = np.sum(mult[:,2])
            class4_correct = np.sum(mult[:,3])
            
            print("Accuracy Results ---------------------------------------")
            print('High Accuracy: ', class1_correct/class1_total)
            print('             Total High: ', class1_total, ' Correct High: ', class1_correct)
            print('CA Accuracy: ', class2_correct/class2_total)
            print('             Total CA : ', class2_total, ' Correct CA : ', class2_correct)
            print('Low Accuracy: ', class3_correct/class3_total)
            print('             Total Low: ', class3_total, ' Correct Low : ', class3_correct)
            print('Healthy Accuracy: ', class4_correct/class4_total)
            print('             Total Healthy : ', class4_total, ' Correct Healthy : ', class4_correct)
            print('*'*20)
            
        elif self.num_classes ==    2:
            class1_total = np.sum(self.flat_val_labels[:,0])
            class2_total = np.sum(self.flat_val_labels[:,1])

            
            class1_correct = np.sum(mult[:,0])
            class2_correct = np.sum(mult[:,1])
            
            print("Accuracy Results ---------------------------------------")
            print('Healthy Accuracy: ', class1_correct/class1_total)
            print('Healthy High: ', class1_total, ' Correct Healthy : ', class1_correct)
            print('Cancer Accuracy: ', class2_correct/class2_total)
            print('Cancer Total: ', class2_total, ' Correct Cancer : ', class2_correct)

            print('*'*20)
            
            
        
        

batch_size = 4**4 # total per batch, must be div by 4, 16 means 4 per class or 8 per class
fc_keep_prob = .9
filters_layer1 = 8
filters_layer2 = 16
filters_layer3 = 32
width1 = 38
width2 = 18
width3 = 16
num_classes= 2 # 4 or 2 only
num_epochs = 40
fc_units = 50
keep_prob=.9
test_every_epoch = True
x_epoch = 15
lr = .001
shuffle=False

# for two class labels, 0 is healthy 1 is cancerous
# For multi Class,
#         self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
# Shifted back 1
#        self.diagnosis_dict = {'high': 0, 'CA': 1, 'low': 2, 'healthy': 3}




net1 = MSInet1(fc_units=fc_units, num_classes=num_classes, width1=width1, width2=width2, width3=width3,filters_layer1=filters_layer1, filters_layer2=filters_layer2, filters_layer3=filters_layer3, batch_size=batch_size)  
net1.buildGraph()
net1.train_X_epoch(lr=lr, keep_prob=keep_prob, num_epochs=num_epochs, test_every_epoch=test_every_epoch, x_epoch=x_epoch,shuffle=shuffle)
