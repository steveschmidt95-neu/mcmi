#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:38:06 2021

@author: stephenschmidt
"""


import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1
v1.disable_eager_execution()
import time
import os
from load_data import H5MSI, SmallROI
from net1_MIL import MSInet1



def single_to_one_hot(labels, num_classes):
        #diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        # shifted to {'high': 0, 'CA': 1, 'low': 2, 'healthy': 3}
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for hot_class in range(0, num_classes):
        class_locations = np.where(labels == hot_class)
        one_hot_labels[class_locations, hot_class] = 1
        
    return(one_hot_labels)


class MIL():
    
    def __init__(self, fc_units = 50, num_classes=4, width1=38, width2=18, width3=16, filters_layer1=12, filters_layer2=24, filters_layer3=48, batch_size=4, lr=.001, keep_prob=.8):
        
        self.smallROI = SmallROI()
        self.smallROI.split_cores()
        self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 0}
        self.batch_size = batch_size
        self.lr = lr
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        
        print("Cores Included: ", self.smallROI.cores_list)
        
        sample_shape = self.smallROI.spec[0].shape[0]
        self.net1 = MSInet1(data_shape = sample_shape, fc_units=fc_units, num_classes=num_classes, width1=width1, width2=width2, width3=width3, filters_layer1=filters_layer1, filters_layer2=filters_layer2, filters_layer3=filters_layer3, batch_size=batch_size,lr=lr)
        self.net1.build_graph()
        
    def init_MSI(self):
        
        self.core_spec = {}
        self.core_true_sub_labels = {}
        self.core_true_label = {}
        self.core_pred_sub_labels = {}
        self.core_probability_labels = {}
        
        for core in self.smallROI.cores_list:
            core_positions = self.smallROI.core_specific_positions[core]
            self.core_spec[core] = self.smallROI.spec[core_positions]
            self.core_true_sub_labels[core] = self.smallROI.subtissue_labels[core_positions]
            self.core_true_label[core] = self.smallROI.tissue_labels[core_positions][0]
            self.core_pred_sub_labels[core] = self.smallROI.tissue_labels[core_positions]
            self.core_probability_labels[core] = self.smallROI.tissue_labels[core_positions]
            
    # Impute the new labels and enforce label constraints
    def enforce_label_constraints(self):
        for core in self.smallROI.cores_list:
            core_positions = self.smallROI.core_specific_positions[core]
            core_label = self.core_true_label[core] = self.smallROI.tissue_labels[core_positions]
            
            # Set all labels to healthy if core is labeled healthy
            if core_label == self.diagnosis_dict['healthy']:
                self.core_pred_labels[:] =  self.diagnosis_dict['healthy']
            
            # Make sure at least one value is labeled non-healthy if core is non-healthy
            else:
                if np.sum():
                    pass   
                

    def compute_params_single_core(self, core):
        
        total_cost = 0
        batch_idx = 0
        total_input_vals = self.core_probability_labels[core].shape[0]
        while (batch_idx+self.batch_size < total_input_vals):
            
            train_batch = self.core_spec[core][batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            train_labels = self.core_pred_sub_labels[core][batch_idx:batch_idx+self.batch_size]
            train_labels = single_to_one_hot(train_labels, self.num_classes)

            
            cost = self.net1.single_tissue_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            total_cost+=cost
            
            batch_idx += self.batch_size
        
        train_batch = self.core_spec[core][total_input_vals-self.batch_size:, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        train_labels = self.core_pred_sub_labels[core][total_input_vals-self.batch_size:]
        train_labels = single_to_one_hot(train_labels, self.num_classes)
        
        cost = self.net1.single_tissue_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
        total_cost+=cost
        
        return(total_cost)

        
        
        
        
    
    def cnn_one_epoch(self):
        pass


MIL = MIL(fc_units = 50, num_classes=4, width1=38, width2=18, width3=16, filters_layer1=12, filters_layer2=24, filters_layer3=48, batch_size=4, lr=.001, keep_prob=.8)
MIL.init_MSI()
MIL.compute_params_single_core(2)
assert False
MIL.enforce_label_constraints()

