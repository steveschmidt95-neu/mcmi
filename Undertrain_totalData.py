#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:12:13 2021

@author: stephenschmidt


uses the undertraining on the total dataset and then applies it to

"""



import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1
v1.disable_eager_execution()
import time
import os
from load_data import H5MSI, SmallROI
from load_train_data import H5MSI_Train
from net1_MIL import MSInet1
import matplotlib.pyplot as plt
import matplotlib
import random


def single_to_one_hot(labels, num_classes):
        #diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        # shifted to {'high': 0, 'CA': 1, 'low': 2, 'healthy': 3}
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for hot_class in range(0, num_classes):
        class_locations = np.where(labels == hot_class)
        one_hot_labels[class_locations, hot_class] = 1
    return(one_hot_labels)

def one_hot_probability_to_single_label(pred_labels, prev_labels, num_classes):
    
    uncertain_pred_locations = np.where(pred_labels[:, 0]==.5)
    
    argmax_labels = np.argmax(pred_labels, axis=1)
    # Dont update lcoations where prediction output was .5
    argmax_labels[uncertain_pred_locations] = prev_labels[uncertain_pred_locations]
    return(argmax_labels)


class MIL():
    
    def __init__(self, fc_units = 50, num_classes=4, width1=38, width2=18, width3=16, filters_layer1=12, 
                 filters_layer2=24, filters_layer3=48, batch_size=4, lr=.001, keep_prob=.8, 
                 small_train=True, undertrain=True):
        
        self.train_data_total = H5MSI_Train()
        self.train_data_total.one_hot_labels()
        self.smallROI = SmallROI(num_classes=num_classes)
        self.smallROI.split_cores()
        
        # Use this for a smaller training set
        if small_train:
            self.smallROI.cores_list = [4, 10, 6, 7]   # One of each 3 non-healthy, one healthy
            #self.smallROI.cores_list = [10, 6, 7,  4,13, 18, 33, 38, 39, 43]  # 1 of each of the 3 non-healthy labels, and all the healthy cores
            #self.smallROI.cores_list = [7, 10,6, 4, 9, 11,33, 34]
            
        
        self.diagnosis_dict =         {'high': 1, 'CA': 2, 'low': 3, 'healthy': 0}
        self.diagnosis_dict_reverse = {1: 'high', 2: 'CA', 3: 'low', 0:'healthy'}
        
        if num_classes == 2:
            self.diagnosis_dict =         {'high': 1, 'healthy': 0}
            self.diagnosis_dict_reverse = {1: 'high',  0:'healthy'}

        self.batch_size = batch_size
        self.lr = lr
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.undertrain=undertrain
        
        print("Cores Included: ", self.smallROI.cores_list)
        
        self.sample_shape = int(self.smallROI.spec[0].shape[0] -2)
        self.net1 = MSInet1(data_shape = self.sample_shape, fc_units=fc_units, num_classes=num_classes, width1=width1, width2=width2, width3=width3, filters_layer1=filters_layer1, filters_layer2=filters_layer2, filters_layer3=filters_layer3, batch_size=batch_size,lr=lr)
        self.net1.build_graph()
        self.highest_score = .5
        self.average_score = 0
        self.final_score = 0
        
        #For Use in undertraining
        self.limiting_factor = 0
        self.positive_predicted = 0
        self.negative_predicted = 0
        self.neg_train_limit_hit = 0
        self.pos_train_limit_hit = 0
    
    # count the number of subtissue labels that will be fed into the training for helathy tissue in fiorst epoch
    def count_healthy_locs_core_only(self):
        total_healthy_subtissue = 0
        for core in self.train_data_total.cores_list:
            if self.train_core_true_label[core] == self.diagnosis_dict['healthy']:
                total_healthy_subtissue+=self.train_core_probability_labels[core].shape[0]
        return(total_healthy_subtissue)
    
        # count the number of subtissue labels that will be fed into the training for helathy tissue in each epoch
    def reset_healthy_subtissue_input_count(self):
        total_healthy_subtissue = 0
        for core in self.train_data.cores_list:
            if self.train_core_true_label[core] == self.diagnosis_dict['healthy']:
                total_healthy_subtissue+=self.train_core_probability_labels[core].shape[0]
            else:
                healthy_predicted_locations = np.where(self.train_core_pred_sub_labels[core] == self.diagnosis_dict['healthy'])[0].shape[0]
                total_healthy_subtissue+=healthy_predicted_locations
        self.total_healthy_subtissue = total_healthy_subtissue
    
        
    def init_MSI(self):
        
        self.train_core_spec = {}
        self.train_core_true_sub_labels = {}
        self.train_core_true_label = {}
        self.train_core_pred_sub_labels = {}
        self.train_core_probability_labels = {}
        
        self.test_core_spec = {}
        self.test_core_true_sub_labels = {}
        self.test_core_true_label = {}
        self.test_core_pred_sub_labels = {}
        self.test_core_probability_labels = {}
        self.test_positions = {}


        
        for core in self.smallROI.cores_list:
            core_positions = self.smallROI.core_specific_positions[core]
            self.test_core_spec[core] = self.smallROI.spec[core_positions]
            self.test_core_true_sub_labels[core] = self.smallROI.subtissue_labels[core_positions]
            self.test_core_true_label[core] = int(self.smallROI.tissue_labels[core_positions][0])
            self.test_core_pred_sub_labels[core] = self.smallROI.tissue_labels[core_positions].astype(int)
            self.test_core_probability_labels[core] = np.zeros((self.smallROI.tissue_labels[core_positions].shape[0], self.num_classes))
            self.test_positions[core] = self.smallROI.position[core_positions].astype(int)
            
            
        for core in self.train_data_total.cores_list:
            if 'Label' not in core:
                self.train_core_spec[core] = self.train_data_total.train_data[core]
                self.train_core_true_label[core] = int(self.train_data_total.train_data[core +'_Labels'][0])
                self.train_core_pred_sub_labels[core] = self.train_data_total.flat_labels[core +'_Labels'].astype(int)
                self.train_core_probability_labels[core] = np.zeros((self.train_data_total.train_data[core].shape[0], self.num_classes))
                
        self.test_total_healthy_subtissue = self.count_healthy_locs_core_only()
    
    
    def _compute_params_healthy_only_single_core2class_hetero(self, core):
        
        total_cost = 0
        batch_idx  = 0
        healthy_computed = 0
        
        valid_input_val_locs = np.where(self.train_core_pred_sub_labels[core]==self.diagnosis_dict['healthy'])[0]
        valid_input_val_locs = list(valid_input_val_locs)
        np.random.shuffle(valid_input_val_locs)
        total_input_vals = len(valid_input_val_locs)
        
        spec = self.train_core_spec[core][valid_input_val_locs]
        
        while (batch_idx+self.batch_size < total_input_vals):
            train_batch = spec[batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            train_labels = np.zeros((self.batch_size, self.num_classes))
            train_labels[:, 0] = 1

            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            total_cost+=cost
            self.train_core_probability_labels[core][batch_idx:batch_idx+self.batch_size] = preds
            
            # Hit the limit on training healthy subtissues
            healthy_computed += batch_size
            if(self.limiting_factor==1 and (healthy_computed > self.positive_predicted)):
                return(total_cost)
            
            batch_idx += self.batch_size
        
        if batch_idx == total_input_vals:
            return(total_cost)
            
        train_batch = spec[((self.batch_size*-1)):, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        train_labels = np.zeros((self.batch_size, self.num_classes))
        train_labels[:, 0] = 1
        
        try:
            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            self.train_core_probability_labels[core][(self.batch_size*-1):, :] = preds
            total_cost+=cost
        except ValueError:
            print("Wring Shape: ", train_batch.shape)
        
        return(total_cost)
    
    def _compute_params_nonhealthy_only_single_core2class_hetero(self, core):
        
        total_cost = 0
        batch_idx  = 0
        non_healthy_computed = 0
        
        valid_input_val_locs = np.where(self.train_core_pred_sub_labels[core]!=self.diagnosis_dict['healthy'])[0]
        valid_input_val_locs = list(valid_input_val_locs)
        np.random.shuffle(valid_input_val_locs)
        total_input_vals = len(valid_input_val_locs)
        
        spec = self.train_core_spec[core][valid_input_val_locs]
        
        while (batch_idx+self.batch_size < total_input_vals):
            train_batch = spec[batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            train_labels = np.zeros((self.batch_size, self.num_classes))
            train_labels[:, 1] = 1

            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            total_cost+=cost
            self.train_core_probability_labels[core][batch_idx:batch_idx+self.batch_size] = preds
            
            # Hit the limit on training non-healthy subtissues
            non_healthy_computed += batch_size
            if(self.limiting_factor==0 and (non_healthy_computed > self.negative_predicted)):
                return(total_cost)
            
            batch_idx += self.batch_size
        
        if batch_idx == total_input_vals:
            return(total_cost)
            
        train_batch = spec[((self.batch_size*-1)):, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        train_labels = np.zeros((self.batch_size, self.num_classes))
        train_labels[:, self.core_true_label[core]] = 1
        
        try:
            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            self.core_probability_labels[core][(self.batch_size*-1):, :] = preds
            total_cost+=cost
        except ValueError:
            print("Wring Shape: ", train_batch.shape)
        
        return(total_cost)
    
    # Heterogenous classes training at the same time
    def _compute_params_single_core_2class(self, core):
        cost = 0
        healthy_computed = 0
        
        total_cost = 0
        batch_idx  = 0
        total_input_vals = self.core_probability_labels[core].shape[0]
        
        while (batch_idx+self.batch_size < total_input_vals):
            
            train_batch = self.train_core_spec[core][batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            train_labels = self.train_core_pred_sub_labels[core][batch_idx:batch_idx+self.batch_size]
            train_labels = single_to_one_hot(train_labels, self.num_classes)

            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            total_cost+=cost
            self.core_probability_labels[core][batch_idx:batch_idx+self.batch_size] = preds
            batch_idx += self.batch_size
            
            healthy_computed += batch_size
            if(self.limiting_factor==1 and (healthy_computed > self.positive_predicted)):
                return(total_cost)
        
        if batch_idx == total_input_vals:
            return(total_cost)
        
        train_batch = self.train_core_spec[core][((self.batch_size)*-1):, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        train_labels = self.train_core_pred_sub_labels[core][(self.batch_size*-1):]
        train_labels = single_to_one_hot(train_labels, self.num_classes)
        
        cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
        self.core_probability_labels[core][(self.batch_size*-1):, :] = preds
        total_cost+=cost
        
        return(total_cost)
    
    # For homogenous tissues
    def _compute_params_single_core_2class_healthy_homogenous(self, core):
        cost = 0
        healthy_computed = 0
        
        total_cost = 0
        batch_idx  = 0
        total_input_vals = self.train_core_probability_labels[core].shape[0]
        
        spec_local = self.train_core_spec[core]
        np.random.shuffle(spec_local)
        
        while (batch_idx+self.batch_size < total_input_vals):
            
            train_batch = spec_local[batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            train_labels = np.zeros((self.batch_size, self.num_classes))
            train_labels[:, self.train_core_true_label[core]] = 1
            
            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            total_cost+=cost
            self.core_probability_labels[core][batch_idx:batch_idx+self.batch_size] = preds
            batch_idx += self.batch_size
            
            healthy_computed += batch_size
            if(self.limiting_factor==1 and (healthy_computed > self.positive_predicted)):
                return(total_cost)
        
        if batch_idx == total_input_vals:
            return(total_cost)
        
        train_batch = spec_local[((self.batch_size)*-1):, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        train_labels = np.zeros((self.batch_size, self.num_classes))
        
        cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
        self.core_probability_labels[core][(self.batch_size*-1):, :] = preds
        total_cost+=cost
        
        return(total_cost)
    
    def compute_params_all_cores2class(self, epoch, balance_training=False, balance_every_x = 5, train_non_healthy_only=True):
        
        epoch_cost = 0
        for core in self.train_data_total.cores_list:
            
            if self.undertrain:
                if self.train_core_true_label[core] == self.diagnosis_dict['healthy']:
                    core_cost = self._compute_params_single_core_2class_healthy_homogenous(core)
                else:
                    non_healthy_cost = self._compute_params_nonhealthy_only_single_core2class_hetero(core)
                    healthy_cost = self._compute_params_healthy_only_single_core2class_hetero(core)
                    core_cost = non_healthy_cost + healthy_cost
            else:
                core_cost = self._compute_params_single_core_2class(core)
            epoch_cost+=core_cost
         
        return(epoch_cost)
            
    def _update_predicted_labels_single_core(self, core, two_class_per_core = True,balance_every_x=5):
        batch_idx = 0
        labels_changed = 0
        total_input_vals = self.train_core_probability_labels[core].shape[0]

        while (batch_idx+self.batch_size < total_input_vals):
            
            train_batch = self.train_core_spec[core][batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            previous_labels = self.train_core_pred_sub_labels[core][batch_idx:batch_idx+self.batch_size]

            preds = self.net1.single_core_predict_labels(train_batch, keep_prob=self.keep_prob)
            new_imputed_labels = one_hot_probability_to_single_label(preds, previous_labels, self.num_classes)
            
            if two_class_per_core:
                new_imputed_labels[np.where(new_imputed_labels!=self.diagnosis_dict['healthy'])] = self.core_true_label[core]
            self.train_core_probability_labels[core][total_input_vals-self.batch_size:] = preds
            
            diffs = len(np.where(previous_labels!=new_imputed_labels)[0])
            
            self.train_core_pred_sub_labels[core][batch_idx:batch_idx+self.batch_size] = new_imputed_labels
            labels_changed+=diffs
            
            batch_idx += self.batch_size

        train_batch = self.train_core_spec[core][total_input_vals-self.batch_size:, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        train_labels = self.train_core_pred_sub_labels[core][total_input_vals-self.batch_size:]
        train_labels = single_to_one_hot(train_labels, self.num_classes)
        
        previous_labels = self.train_core_pred_sub_labels[core][total_input_vals-self.batch_size:]
    
        preds = self.net1.single_core_predict_labels(train_batch, keep_prob=self.keep_prob)
        new_imputed_labels = one_hot_probability_to_single_label(preds, previous_labels, self.num_classes)
        if two_class_per_core:
            new_imputed_labels[np.where(new_imputed_labels!=self.diagnosis_dict['healthy'])] = self.core_true_label[core]
        self.train_core_probability_labels[core][total_input_vals-self.batch_size:] = preds
        
        diffs = len(np.where(previous_labels!=new_imputed_labels)[0])
        self.train_core_pred_sub_labels[core][total_input_vals-self.batch_size:] = new_imputed_labels
        labels_changed+=diffs
        
        return(labels_changed)
        
    def get_test_labels_single_core(self, core):
        batch_idx = 0
        labels_changed = 0
        total_input_vals = self.test_core_probability_labels[core].shape[0]

        while (batch_idx+self.batch_size < total_input_vals):
            
            train_batch = self.test_core_spec[core][batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            previous_labels = self.test_core_pred_sub_labels[core][batch_idx:batch_idx+self.batch_size]

            preds = self.net1.single_core_predict_labels(train_batch, keep_prob=self.keep_prob)
            new_imputed_labels = one_hot_probability_to_single_label(preds, previous_labels, self.num_classes)
            
            if two_class_per_core:
                new_imputed_labels[np.where(new_imputed_labels!=self.diagnosis_dict['healthy'])] = self.test_core_true_label[core]
            self.test_core_probability_labels[core][total_input_vals-self.batch_size:] = preds
            
            diffs = len(np.where(previous_labels!=new_imputed_labels)[0])
            
            self.test_core_pred_sub_labels[core][batch_idx:batch_idx+self.batch_size] = new_imputed_labels
            labels_changed+=diffs
            
            batch_idx += self.batch_size

        train_batch = self.test_core_spec[core][total_input_vals-self.batch_size:, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        train_labels = self.test_core_pred_sub_labels[core][total_input_vals-self.batch_size:]
        train_labels = single_to_one_hot(train_labels, self.num_classes)
        
        previous_labels = self.test_core_pred_sub_labels[core][total_input_vals-self.batch_size:]
    
        preds = self.net1.single_core_predict_labels(train_batch, keep_prob=self.keep_prob)
        new_imputed_labels = one_hot_probability_to_single_label(preds, previous_labels, self.num_classes)
        if two_class_per_core:
            new_imputed_labels[np.where(new_imputed_labels!=self.diagnosis_dict['healthy'])] = self.core_true_label[core]
        self.test_core_probability_labels[core][total_input_vals-self.batch_size:] = preds
        
        diffs = len(np.where(previous_labels!=new_imputed_labels)[0])
        self.test_core_pred_sub_labels[core][total_input_vals-self.batch_size:] = new_imputed_labels
        labels_changed+=diffs
        
        
        
    
    def impute_labels_all_cores(self,two_class_per_core=False):
                
        labels_changed = 0
        for core in self.train_data_total.cores_list:
            # dont impute healthy cores
            if self.train_core_true_label[core] == self.diagnosis_dict['healthy']:
                continue
            core_labels_changed = self._update_predicted_labels_single_core(core,two_class_per_core)
            labels_changed+=core_labels_changed
        return(labels_changed)
    
    def count_predicted_labels2class(self):
        limiting_factor = 0
        positive_predicted_total = 0
        healthy_predicted_total = 0
        total_count = 0
        
        for core in self.train_data_total.cores_list:
            
            total_subtissues = self.train_core_pred_sub_labels[core].shape[0]
            total_count += total_subtissues

            positive_predicted = (len(np.where(self.train_core_pred_sub_labels[core] == 1)[0]))
            healthy_predicted = int(total_subtissues - positive_predicted)
            
            positive_predicted_total+=positive_predicted
            healthy_predicted_total+=healthy_predicted
            
        if positive_predicted_total < healthy_predicted_total:
            limiting_factor= 1
        
        print("Limiting Factor: ", limiting_factor)
        assert (healthy_predicted_total + positive_predicted_total) == total_count
        self.healthy_predicted = healthy_predicted_total
        self.positive_predicted = positive_predicted_total
        self.limiting_factor = limiting_factor
        
    def count_predicted_labels2class_print(self):
        positive_predicted_total = 0
        healthy_predicted_total = 0
        total_count = 0
        
        for core in self.train_data_total.cores_list:
            #print("COre: ", core, "Label: ", self.core_true_label[core])
            total_subtissues = self.train_core_pred_sub_labels[core].shape[0]
            total_count += total_subtissues

            positive_predicted = (len(np.where(self.train_core_pred_sub_labels[core] == 1)[0]))
            healthy_predicted = int(total_subtissues - positive_predicted)            
            
            positive_predicted_total+=positive_predicted
            healthy_predicted_total+=healthy_predicted
            
        
        print('Param Computing Tissue Count --')
        print("Total Subtissues: ", total_count)
        print("Healthy Count: ", healthy_predicted_total)
        print("High Count: ", positive_predicted_total)
        
    
    
    def cnn_X_epoch(self, x, balance_classes=False, reset_healthy_count=False,balance_every_x=5,test_every_x=5, two_class_per_core=False, train_non_healthy_only=False, enforce_healthy_constraint=False, shuffle_core_list=True):
        self.count_predicted_labels2class_print()
        for epoch in range(1, x):
            print("Epoch ", epoch)
            if shuffle_core_list:
                print("Shuffling Cores List")
                random.shuffle(self.train_data_total.cores_list)   
            if self.undertrain:
                self.count_predicted_labels2class()
            if self.num_classes == 2:
                cost = self.compute_params_all_cores2class(epoch, balance_training=balance_classes, balance_every_x=balance_every_x,train_non_healthy_only=train_non_healthy_only)
            else:
                cost = self.compute_params_all_cores(epoch, balance_training=balance_classes, balance_every_x=balance_every_x,train_non_healthy_only=train_non_healthy_only)
            
            self.count_predicted_labels2class_print()
            print('    Cost: ', cost)            
            labels_changed = self.impute_labels_all_cores(two_class_per_core=two_class_per_core)
            print("    Labels Changed: ", labels_changed)
            self.enforce_label_constraints(enforce_healthy_constraint=enforce_healthy_constraint)
            if epoch % test_every_x == 0:
                if self.num_classes == 2:
                    self.eval_all_cores2class()
                else:
                    self.eval_all_cores()
            if reset_healthy_count:
                self.reset_healthy_subtissue_input_count()
        print('Highest Score: ', self.highest_score)
        print('Final Score: ', self.final_score)
        print('Average Score: ', self.average_score/x)
                
                
    # Impute the new labels and enforce label constraints
    def enforce_label_constraints(self, enforce_healthy_constraint=False):
        k_large_elements = self.batch_size
        for core in self.train_data_total.cores:
            core_label = self.train_core_true_label[core]
            
            # Set all labels to healthy if core is labeled healthy
            if core_label == self.diagnosis_dict['healthy']:
                self.train_core_pred_sub_labels[core][:] =  self.diagnosis_dict['healthy']
            
            # Make sure at least k values is labeled non-healthy if core is non-healthy
            else:
                # All labels are healthy
                if np.sum(self.train_core_pred_sub_labels[core]) <= k_large_elements or np.where(self.train_core_pred_sub_labels[core] == core_label)[0].shape[0] == 0:
                    #Pick the max values specific to the core class
                    k_max_element_lcoations = np.argpartition(self.core_probability_labels[core][:, core_label], -k_large_elements)[-k_large_elements:]
                    self.train_core_pred_sub_labels[core][k_max_element_lcoations] = core_label
                    
                # apply the same rule to healthy tissue, not leaving a tissue with comptely unhealthy labels
                if enforce_healthy_constraint:
                    healthy_locations = np.where(self.train_core_pred_sub_labels[core] == self.diagnosis_dict['healthy'])
                    healthy_count = healthy_locations[0].shape[0]

                    if healthy_count <k_large_elements:
                        k_max_element_lcoations = np.argpartition(self.train_core_probability_labels[core][:, self.diagnosis_dict['healthy']], -k_large_elements)[-k_large_elements:]
                        self.train_core_pred_sub_labels[core][k_max_element_lcoations] = self.diagnosis_dict['healthy']
    
    # Evaulte according to only the given subtissue labels
    def eval_single_core_direct_labels(self, core):
        true_labels = self.test_core_true_sub_labels[core]
        pred_labels = self.test_core_pred_sub_labels[core]
        core_label = self.test_core_true_label[core]
        
        # Case for healthy tissues
        if(core_label == self.diagnosis_dict['healthy']):
            tn = pred_labels.shape[0] - np.sum(pred_labels)
            total_neg = pred_labels.shape[0]
            total_pos = 0
            tp = 0
        
        # Case for non-healthy tissues
        else:
            total_pos = np.where(true_labels == core_label)[0].shape[0]
            total_neg = np.where(true_labels != core_label)[0].shape[0]
            pos_locations = np.where((true_labels == core_label))
            neg_locations = np.where((true_labels != core_label))
            
            expected_core_postive_locations = pred_labels[pos_locations]
            expected_core_negative_locations = pred_labels[neg_locations]
            
            tp = np.where(expected_core_postive_locations == core_label)[0].shape[0]
            tn = np.where(expected_core_negative_locations == self.diagnosis_dict['healthy'])[0].shape[0]
        
        return(tp, tn, total_pos, total_neg, core_label)
        
    def eval_all_cores2class(self):
        # THis is after enforcing constraints
        #self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 0}
        tp_high = 0
        tn_all_cores = 0
        total_high = 0
        total_neg_all_cores = 0
        
        for core in self.smallROI.cores_list:
            if self.test_core_true_label[core] == self.diagnosis_dict['healthy']:
                continue
            
            else:
                self.get_test_labels(core)
            
            tp, tn, total_pos, total_neg, core_label = self.eval_single_core_direct_labels(core)
            
            if core_label == self.diagnosis_dict['healthy']:
                tn_all_cores += tn
                total_neg_all_cores += total_neg
                
            elif core_label == self.diagnosis_dict['high']:
                tp_high += tp
                total_high += total_pos
                tn_all_cores += tn
                total_neg_all_cores += total_neg
                
        print('  - Evaluating All Cores - ')
        accuracy = (tp_high + tn_all_cores) / (total_high + total_neg_all_cores)
        print('Accuracy: ', accuracy)
        
        if total_neg_all_cores == 0:
            neg_accuracy = -1
        else:
            neg_accuracy = (tn_all_cores / total_neg_all_cores)
        if total_high == 0:
            high_accuracy = -1
        else:
            high_accuracy = (tp_high / total_high)
        
        print('Neg Accuracy: ', neg_accuracy)
        print('High Accuracy: ', high_accuracy)
        
        balanced_accuracy = (neg_accuracy  * .5) + (high_accuracy*.5) 
        print('Balanced Accuracy: ', balanced_accuracy)
        self.final_score = balanced_accuracy
        self.average_score += balanced_accuracy
        
        if balanced_accuracy > self.highest_score:
            self.highest_score = balanced_accuracy
        print('Highest Balanced Accuracy: ', self.highest_score)
        print('  - Resuming Training  - ')
    
    
   
    def viz_single_core_pred(self, core):
        positions = self.test_positions[core]
        
        xmax = np.max(positions[:, 0])
        xmin = np.min(positions[:, 0])
        ymax = np.max(positions[:, 1])
        ymin = np.min(positions[:, 1])
        image_array = np.zeros((xmax-xmin+1, ymax-ymin+1))
        image_array[:] = 4
        
        cmap = matplotlib.colors.ListedColormap(['black', 'red', 'blue', 'yellow', 'white'])
        bounds = [0, 1, 2, 3, 4,5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        for location in range(0, self.test_core_pred_sub_labels[core].shape[0]):
            label = self.test_core_pred_sub_labels[core][location]
            xloc = self.test_positions[core][location][0]- xmin
            yloc = self.test_positions[core][location][1] - ymin
            
            image_array[xloc, yloc] = label
        
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.grid(True)
        plt.imshow(image_array, interpolation='nearest',cmap=cmap,norm=norm)

        title = "Core Number: " + str(core)  + " Label: " +  self.diagnosis_dict_reverse[self.core_true_label[core]]
        print(title)
        plt.title(title)
        plt.colorbar(cmap=cmap,norm=norm,boundaries=bounds,ticks=[0,1,2, 3])
        filename = 'Images/Pred'+ str(int(core)) + '.png'
        print(filename)
        
        plt.savefig(filename, pad_inches=0)
        plt.clf()
    
    def save_ims_all_cores(self):
        for core in self.smallROI.cores_list:
            self.viz_single_core_pred(core)
    
batch_size = 8
balance_classes=False # train on same amount of each class per epoch
balance_every_x = 1
two_class_per_core=False # make labels in each core only be healthy or the core label
reset_healthy_count = False # include healthy subtissue in non-healthy cores to count used for balancing classes
train_non_healthy_only = False # train on only the non-healthy assigned locations in non-healthy tissues
enforce_healthy_constraint = True # Enforce the same constraint for healthy tissus on non-healthy cores
small_train = False
shuffle_core_list = True
undertrain=True

test_every_x = 3
num_epochs=150
lr=.001

MIL = MIL(fc_units = 100, num_classes=2, width1=38,  width2=18, width3=16, filters_layer1=40, 
    filters_layer2=60, filters_layer3=100, batch_size=batch_size, lr=lr, keep_prob=.99,
    small_train=small_train,undertrain=undertrain)
MIL.init_MSI()
MIL.cnn_X_epoch(num_epochs,balance_classes=balance_classes,reset_healthy_count=reset_healthy_count, 
                balance_every_x=balance_every_x, test_every_x=test_every_x,
                two_class_per_core=two_class_per_core, train_non_healthy_only=train_non_healthy_only,
                enforce_healthy_constraint=enforce_healthy_constraint, shuffle_core_list=shuffle_core_list)
