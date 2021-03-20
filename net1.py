#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:55:26 2021

@author: stephenschmidt
"""


import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1
v1.disable_eager_execution()
import time
import os
from load_data import H5MSI, shuffle_data


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
        msi_dataset.flatten_data()
        
        if num_classes == 2:
            msi_dataset.two_class_data() # remove this for multilable training
        
        msi_dataset.split_data()
        
        self.flat_train = msi_dataset.split_train
        self.flat_train_labels = msi_dataset.split_train_labels
        self.flat_val = msi_dataset.split_val
        self.flat_val_labels =  msi_dataset.split_val_labels
        self.in_shape = self.flat_train.shape[1]
        self.fc_unit = fc_units
        self.batch_size = batch_size
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
        
        W_conv1 = weight_variable('W_conv1',[width1, 1, filters_layer1])
        b_conv1 = bias_variable('b_conv1',[filters_layer1])
        
        W_conv2 = weight_variable('W_conv1_2',[width2, filters_layer1, filters_layer2])
        b_conv2 = bias_variable('b_conv1_2',[filters_layer2])
         
        W_conv3 = weight_variable('W_conv1_3',[width3, filters_layer2, filters_layer3])
        b_conv3 = bias_variable('b_conv1_3',[filters_layer3])

        
        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        conv1 =   tf.nn.relu(conv1d(self.xdev, W_conv1) + b_conv1) 
        pool1 = tf.nn.max_pool(conv1, 2, 2, padding= "VALID")
        
        batch1 = v1.layers.batch_normalization(pool1, training=self.training)

        conv2 = tf.nn.relu(conv1d(batch1, W_conv2) + b_conv2) 
        pool2 = tf.nn.max_pool(conv2, 2, 2, padding= "VALID")
        
        batch2 = v1.layers.batch_normalization(pool2, training=self.training)
        
        conv3 =   tf.nn.relu(conv1d(batch2, W_conv3) + b_conv3) 
        pool3 = tf.nn.max_pool(conv3, 2, 2, padding="VALID")
        flat_final_conv = v1.layers.flatten(pool3)
        
        W_fc1 = weight_variable('W_fc1', [flat_final_conv.shape[1], fc_units])
        b_fc1 = bias_variable('b_fc1',  [fc_units])
        
        W_fc2 = weight_variable('W_fc2', [fc_units, self.num_classes])
        b_fc2 = bias_variable('b_fc2',  [self.num_classes])
        
        fc1 = tf.nn.relu(tf.matmul(flat_final_conv, W_fc1) + b_fc1)
        #drop_fc1 = tf.nn.dropout(fc1, self.fc_keep_prob, training=self.training)
        drop_fc1 = v1.layers.dropout(fc1, self.fc_keep_prob, training=self.training)
        
        batch3 = v1.layers.batch_normalization(drop_fc1, training=self.training)

        fc2 = tf.nn.relu(tf.matmul(batch3, W_fc2) + b_fc2)
        
        self.y_conv = tf.identity(fc2, name='full_op')
        
    def get_next_batch(self):
        shuffled_data, shuffled_labels = shuffle_data(self.flat_train, self.flat_train_labels, self.num_classes)
        batch_div = self.batch_size//self.num_classes
        train_batch = np.zeros((self.batch_size, self.flat_train.shape[1]))
        train_labels = np.zeros((self.batch_size, self.num_classes))
        
        for class_idx in range(0,self.num_classes):
            class_locs = np.where(shuffled_labels[:, class_idx]==1)[0:batch_div]
            assert len(class_locs[0]) >= batch_div # Need 
            class_locs = class_locs[0][0:batch_div]
            class_values = shuffled_data[class_locs[0:batch_div], :]
            class_labels = shuffled_labels[class_locs, :]
            
            train_batch[((self.batch_size//self.num_classes)*class_idx):(batch_div)*class_idx +batch_div, :] = class_values
            train_labels[((self.batch_size//self.num_classes)*class_idx):(batch_div)*class_idx +batch_div, :] = class_labels
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        return(train_batch, train_labels)
    
    def get_test_batch(self):
        batch_div = self.batch_size//self.num_classes
        shuffled_data, shuffled_labels = shuffle_data(self.flat_val, self.flat_val_labels, self.num_classes)
        val_batch = np.zeros((self.batch_size, self.flat_val.shape[1]))
        val_labels = np.zeros((self.batch_size, self.num_classes))
        
        for class_idx in range(0,self.num_classes):
            class_locs = np.where(shuffled_labels[:, class_idx]==1)[0:batch_div]
            assert len(class_locs[0]) >= batch_div # Need 
            class_locs = class_locs[0][0:batch_div]
            class_values = shuffled_data[class_locs[0:batch_div], :]
            class_labels = shuffled_labels[class_locs, :]
            
            val_batch[((self.batch_size//self.num_classes)*class_idx):(batch_div)*class_idx +batch_div, :] = class_values
            val_labels[((self.batch_size//self.num_classes)*class_idx):(batch_div)*class_idx +batch_div, :] = class_labels
        val_batch = np.reshape(val_batch, (val_batch.shape[0], val_batch.shape[1], 1))

        return(val_batch, val_labels)
        
        
        
    def train_X_epoch(self, num_epochs = 1, lr = .001, keep_prob=.8,  test_every_epoch = False, x_epoch=5):
    
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
            classification = self.sess.run(v1.nn.softmax(self.y_conv), feed_dict)


            epoch_loss_sum += cross_entropy_py
                
            sci_loss = '%e' % epoch_loss_sum
            print('')
            print('Epoch ' + str(epoch+1) + ' Loss: ' + str(sci_loss))
            

            if test_every_epoch and ((epoch+1)%x_epoch == 0) and epoch !=0:
                '''
               # This is for testing all validation data, pretty slow
                
                batch_idx = 0
                label_predictions = np.zeros_like(self.flat_val_labels)

                while (batch_idx < self.flat_val.shape[0] and batch_idx+self.batch_size < self.flat_val.shape[0]):

                    val_batch = self.flat_val[batch_idx:batch_idx+self.batch_size,:]
                    val_batch = np.reshape(val_batch, (val_batch.shape[0], val_batch.shape[1], 1))
                    
                    feed_dict={self.xdev: val_batch, self.fc_keep_prob: keep_prob, self.training: False }

                    classification = self.sess.run(v1.nn.softmax(self.y_conv), feed_dict)
                    label_predictions[batch_idx:batch_idx+self.batch_size, :] = classification
                    batch_idx += self.batch_size
                    
                    self.metric_validation(label_predictions)
                    assert False
                    
                
                remaining = (self.flat_val.shape[0] - batch_idx)
                if (remaining > 0):
                    val_batch = np.zeros((self.batch_size, self.flat_val.shape[1]))
                    val_batch[0:remaining, :] = self.flat_val[batch_idx:batch_idx+remaining, :]
                val_batch = np.reshape(val_batch, (val_batch.shape[0], val_batch.shape[1], 1))
                feed_dict={self.xdev: val_batch, self.fc_keep_prob: keep_prob, self.training: False}
                classification = self.sess.run(v1.nn.softmax(self.y_conv), feed_dict)
                label_predictions[batch_idx:batch_idx+remaining, :] = classification[0:remaining, :]
                '''
                val_data, val_labels = self.get_test_batch()
                feed_dict={self.xdev: val_data, self.fc_keep_prob: keep_prob, self.training: False }
                classification = self.sess.run(v1.nn.softmax(self.y_conv), feed_dict)

                self.metric_validation(classification, batch_labels=val_labels)


                
    def metric_validation(self, predictions, batch_labels=None):
        one_hot_predictions= np.zeros_like(predictions)
        one_hot_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1
        if batch_labels is not None:
            mult = batch_labels * one_hot_predictions
        else:
            mult = self.flat_val_labels * one_hot_predictions
        print('')
        
        
        if self.num_classes == 4:
            
            if batch_labels is not None:
                class1_total = np.sum(batch_labels[:,0])
                class2_total = np.sum(batch_labels[:,1])
                class3_total = np.sum(batch_labels[:,2])
                class4_total = np.sum(batch_labels[:,3])
            else:
                class1_total = np.sum(self.flat_val_labels[:,0])
                class2_total = np.sum(self.flat_val_labels[:,1])
                class3_total = np.sum(self.flat_val_labels[:,2])
                class4_total = np.sum(self.flat_val_labels[:,3])

            
            class1_correct = np.sum(mult[:,0])
            class2_correct = np.sum(mult[:,1])
            class3_correct = np.sum(mult[:,2])
            class4_correct = np.sum(mult[:,3])
            
            print("Accuracy Results ---------------------------------------")
            print('High Accuracy: ', (class1_correct/class1_total)*100, '%')
            
            print('CA Accuracy: ', (class2_correct/class2_total)*100, '%')
            
            print('Low Accuracy: ', (class3_correct/class3_total)*100, '%')
           
            print('Healthy Accuracy: ', (class4_correct/class4_total)*100, '%')
            print('             Total High: ', class1_total, ' Correct High: ', class1_correct)
            print('             Total CA : ', class2_total, ' Correct CA : ', class2_correct)
            print('             Total Low: ', class3_total, ' Correct Low : ', class3_correct)
            print('             Total Healthy : ', class4_total, ' Correct Healthy : ', class4_correct)
            print('*'*20)
            
        elif self.num_classes ==    2:
            
            if batch_labels is not None:
                class1_total = np.sum(batch_labels[:,0])
                class2_total = np.sum(batch_labels[:,1])
            else:
                    
                class1_total = np.sum(self.flat_val_labels[:,0])
                class2_total = np.sum(self.flat_val_labels[:,1])
            
            class1_correct = np.sum(mult[:,0])
            class2_correct = np.sum(mult[:,1])
            
            print("Accuracy Results ---------------------------------------")
            print('Healthy Accuracy: ', (class1_correct/class1_total)*100, '%')
            
            print('Cancer Accuracy: ', (class2_correct/class2_total)*100, '%')
            print('                  Healthy Total: ', class1_total, ' Correct Healthy : ', class1_correct)
            print('                 Cancer Total: ', class2_total, ' Correct Cancer : ', class2_correct)
            print('*'*20)
        

batch_size = 4**4# total per batch, must be div by 4, 16 means 4 per class or 8 per class
filters_layer1 = 16
filters_layer2 = 32
filters_layer3 = 64
width1 = 38
width2 = 18
width3 = 16
num_classes= 2 # 4 or 2 only
num_epochs = 1000
fc_units = 100
keep_prob=.75
test_every_epoch = True
x_epoch = 10
lr = .001

# for two class labels, 0 is healthy 1 is cancerous
# For multi Class,
#         self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
# Shifted back 1
#        self.diagnosis_dict = {'high': 0, 'CA': 1, 'low': 2, 'healthy': 3}




net1 = MSInet1(fc_units=fc_units, num_classes=num_classes, width1=width1, width2=width2, width3=width3,filters_layer1=filters_layer1, filters_layer2=filters_layer2, filters_layer3=filters_layer3, batch_size=batch_size)  
net1.buildGraph()
net1.train_X_epoch(lr=lr, keep_prob=keep_prob, num_epochs=num_epochs, test_every_epoch=test_every_epoch, x_epoch=x_epoch)
