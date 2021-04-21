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
import matplotlib.pyplot as plt
import matplotlib



def single_to_one_hot(labels, num_classes):
        #diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        # shifted to {'high': 0, 'CA': 1, 'low': 2, 'healthy': 3}
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for hot_class in range(0, num_classes):
        class_locations = np.where(labels == hot_class)
        one_hot_labels[class_locations, hot_class] = 1
    return(one_hot_labels)

def one_hot_probability_to_single_label(labels, num_classes):
    argmax_labels = np.argmax(labels, axis=1)
    return(argmax_labels)


class MIL():
    
    def __init__(self, fc_units = 50, num_classes=4, width1=38, width2=18, width3=16, filters_layer1=12, filters_layer2=24, filters_layer3=48, batch_size=4, lr=.001, keep_prob=.8, small_train=True):
        
        self.smallROI = SmallROI()
        self.smallROI.split_cores()
        
        # Use this for a smaller training set
        if small_train:
            self.smallROI.cores_list = [7, 10,6, 4,]
        
        self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 0}
        self.batch_size = batch_size
        self.lr = lr
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        
        print("Cores Included: ", self.smallROI.cores_list)
        
        self.sample_shape = int(self.smallROI.spec[0].shape[0])
        self.net1 = MSInet1(data_shape = self.sample_shape, fc_units=fc_units, num_classes=num_classes, width1=width1, width2=width2, width3=width3, filters_layer1=filters_layer1, filters_layer2=filters_layer2, filters_layer3=filters_layer3, batch_size=batch_size,lr=lr)
        self.net1.build_graph()
    
    # count the number of subtissue labels that will be fed into the training for helathy tissue in fiorst epoch
    def count_healthy_locs_core_only(self):
        total_healthy_subtissue = 0
        for core in self.smallROI.cores_list:
            if self.core_true_label[core] == self.diagnosis_dict['healthy']:
                total_healthy_subtissue+=self.core_probability_labels[core].shape[0]
        return(total_healthy_subtissue)
    
        # count the number of subtissue labels that will be fed into the training for helathy tissue in each epoch
    def reset_healthy_subtissue_input_count(self):
        total_healthy_subtissue = 0
        for core in self.smallROI.cores_list:
            if self.core_true_label[core] == self.diagnosis_dict['healthy']:
                total_healthy_subtissue+=self.core_probability_labels[core].shape[0]
            else:
                healthy_predicted_locations = np.where(self.core_pred_sub_labels[core] == self.diagnosis_dict['healthy'])[0].shape[0]
                total_healthy_subtissue+=healthy_predicted_locations
        self.total_healthy_subtissue = total_healthy_subtissue
    
        
    def init_MSI(self):
        
        self.core_spec = {}
        self.core_true_sub_labels = {}
        self.core_true_label = {}
        self.core_pred_sub_labels = {}
        self.core_probability_labels = {}
        self.positions = {}

        
        for core in self.smallROI.cores_list:
            core_positions = self.smallROI.core_specific_positions[core]
            self.core_spec[core] = self.smallROI.spec[core_positions]
            self.core_true_sub_labels[core] = self.smallROI.subtissue_labels[core_positions]
            self.core_true_label[core] = int(self.smallROI.tissue_labels[core_positions][0])
            self.core_pred_sub_labels[core] = self.smallROI.tissue_labels[core_positions].astype(int)
            self.core_probability_labels[core] = np.zeros((self.smallROI.tissue_labels[core_positions].shape[0], self.num_classes))
            self.positions[core] = self.smallROI.position[core_positions].astype(int)

            
        self.total_healthy_subtissue = self.count_healthy_locs_core_only()
    
    def _compute_params_nonhealthy_only_single_core(self, core):
        tissue_count = {'high': 0, 'CA': 0, 'low': 0, 'healthy': 0}
        
        total_cost = 0
        batch_idx  = 0
        
        valid_input_val_locs = np.where(self.core_pred_sub_labels[core]!=self.diagnosis_dict['healthy'])
        total_input_vals = valid_input_val_locs[0].shape[0]
        
        if total_input_vals < self.batch_size:
            print('Total Onput: ', total_input_vals)
        
        spec = self.core_spec[core][valid_input_val_locs]
        
        
        while (batch_idx+self.batch_size < total_input_vals):
            train_batch = spec[batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            train_labels = np.zeros((self.batch_size, 4))
            train_labels[:, self.core_true_label[core]] = 1
            
            tissue_count['high'] += np.sum(train_labels[:, self.diagnosis_dict['high']])
            tissue_count['low'] += np.sum(train_labels[:, self.diagnosis_dict['low']])
            tissue_count['CA'] += np.sum(train_labels[:, self.diagnosis_dict['CA']])
            tissue_count['healthy'] += np.sum(train_labels[:, self.diagnosis_dict['healthy']])


            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            total_cost+=cost
            self.core_probability_labels[core][batch_idx:batch_idx+self.batch_size] = preds
            
            batch_idx += self.batch_size
        
        if batch_idx == total_input_vals:
            return(total_cost, tissue_count)
            
        train_batch = spec[(self.batch_size*-1):, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        try:
            assert train_batch.shape[0] == self.batch_size
        except AssertionError:
            print('Wrong Shape')
            print(train_batch.shape)
            return(total_cost, tissue_count)
        
        train_labels = np.zeros((self.batch_size, 4))
        train_labels[:, self.core_true_label[core]] = 1
        
        tissue_count['high'] += np.sum(train_labels[:, self.diagnosis_dict['high']])
        tissue_count['low'] += np.sum(train_labels[:, self.diagnosis_dict['low']])
        tissue_count['CA'] += np.sum(train_labels[:, self.diagnosis_dict['CA']])
        tissue_count['healthy'] += np.sum(train_labels[:, self.diagnosis_dict['healthy']])
        
        cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
        self.core_probability_labels[core][(self.batch_size*-1):, :] = preds
        total_cost+=cost
        
        return(total_cost, tissue_count)
        
    
    def _compute_params_single_core(self, core):
        
        tissue_count = {'high': 0, 'CA': 0, 'low': 0, 'healthy': 0}
        
        total_cost = 0
        batch_idx  = 0
        total_input_vals = self.core_probability_labels[core].shape[0]
        while (batch_idx+self.batch_size < total_input_vals):
            
            train_batch = self.core_spec[core][batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            train_labels = self.core_pred_sub_labels[core][batch_idx:batch_idx+self.batch_size]
            train_labels = single_to_one_hot(train_labels, self.num_classes)
            
            tissue_count['high'] += np.sum(train_labels[:, self.diagnosis_dict['high']])
            tissue_count['low'] += np.sum(train_labels[:, self.diagnosis_dict['low']])
            tissue_count['CA'] += np.sum(train_labels[:, self.diagnosis_dict['CA']])
            tissue_count['healthy'] += np.sum(train_labels[:, self.diagnosis_dict['healthy']])

            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            total_cost+=cost
            self.core_probability_labels[core][batch_idx:batch_idx+self.batch_size] = preds
            batch_idx += self.batch_size
        
        if batch_idx == total_input_vals:
            return(total_cost, tissue_count)
        
        train_batch = self.core_spec[core][(self.batch_size*-1):, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        try:
            assert train_batch.shape[0] == self.batch_size
        except AssertionError:
            print('Wrong Shape')
            print(train_batch.shape)
            return(total_cost, tissue_count)
        
        train_labels = self.core_pred_sub_labels[core][(self.batch_size*-1):]
        train_labels = single_to_one_hot(train_labels, self.num_classes)
        
        tissue_count['high'] += np.sum(train_labels[:, self.diagnosis_dict['high']])
        tissue_count['low'] += np.sum(train_labels[:, self.diagnosis_dict['low']])
        tissue_count['CA'] += np.sum(train_labels[:, self.diagnosis_dict['CA']])
        tissue_count['healthy'] += np.sum(train_labels[:, self.diagnosis_dict['healthy']])
        
        cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
        self.core_probability_labels[core][(self.batch_size*-1):, :] = preds
        total_cost+=cost
        
        return(total_cost, tissue_count)
    
    # Makes sure you are training on the same number of inputs per class in each epoch
    def balance_training(self, total_tissue_count):
        balance_classes = {'low': 0, 'high': 0, 'CA': 0}
        total_cost = 0
        
        for balance_class in balance_classes.keys():
            k= int(self.total_healthy_subtissue - total_tissue_count[balance_class])
            
            # no need to balance classes with enough labels
            if k <= 0:
                print('No need to balance ', balance_class)
                continue
            
            k_highest = self.choose_k_highest_probs_from_class(k, balance_class)
            k_count = k_highest.shape[0]
                
            # make sure you have enough labels to fit into batch
            # Double array repeatdly until its large enough
        
            # make sure you have enough from the request
            while (k >= k_count):
                if k_count == 1:
                    doubled_array = np.zeros((k_count+1, k_highest.shape[1]))
                    doubled_array[0, :] = k_highest
                    doubled_array[1, :] = k_highest
                else:
                    doubled_array = np.zeros((k_count*2, k_highest.shape[1]))
                    doubled_array[0:k_highest.shape[0]] = k_highest
                    doubled_array[k_highest.shape[0]:] = k_highest
                
                k_count = doubled_array.shape[0]
                k_highest = doubled_array
            
            # make sure you have enough to fit the batch size
            while (self.batch_size >= k_count):
                if k_count == 1:
                    doubled_array = np.zeros((k_count+1, k_highest.shape[1]))
                    doubled_array[0, :] = k_highest
                    doubled_array[1, :] = k_highest
                else:
                    doubled_array = np.zeros((k_count*2, k_highest.shape[1]))
                    doubled_array[0:k_highest.shape[0]] = k_highest
                    doubled_array[k_highest.shape[0]:] = k_highest
                
                k_count = doubled_array.shape[0]
                k_highest = doubled_array
                
            k_highest = k_highest[0:k, :]
            k_count = k_highest.shape[0]
            
            print('Balancing ', balance_class, ' By ', k_count, ' Inputs')
            batch_idx = 0
            while (batch_idx+self.batch_size < k_count):
                train_batch = k_highest[batch_idx:batch_idx+self.batch_size, :]
                train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))

                train_labels = np.zeros((self.batch_size, self.num_classes))
                train_labels[:, self.diagnosis_dict[balance_class]] = 1
                
                cost, _ = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
                total_cost+=cost
                batch_idx+= self.batch_size
            
            train_batch = k_highest[k_count-self.batch_size:, :]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            train_labels = np.zeros((self.batch_size, self.num_classes))
            train_labels[:, self.diagnosis_dict[balance_class]] = 1

        return(cost)
    
    # Gets the k most likely tissues for a specific class from the subtissue labels that were predicted to be that class
    def choose_k_highest_probs_from_class(self, k, class_needed):
        total_class_specific_predicted_count = 0
        
        # get total number of values predicted to be this class
        for core in self.smallROI.cores_list:
            class_pred_locations = np.where(self.core_pred_sub_labels[core] == self.diagnosis_dict[class_needed])                
            total_class_specific_predicted_count += class_pred_locations[0].shape[0]
            
            # If there are none to be predicted in this cores for a certain class and its core label is the class,
            # choose the highest values from each core with that label
            if self.core_true_label[core] == class_needed and class_pred_locations[0].shape[0] == 0:
                total_class_specific_predicted_count+=1
        
        k_highest_values = np.zeros((k, self.sample_shape))
        all_predicted_probability_values = np.zeros((total_class_specific_predicted_count))
        all_pred_mspec_vals = np.zeros((total_class_specific_predicted_count, self.sample_shape))
        
        loc = 0
        for core in self.smallROI.cores_list:
            class_pred_locations = np.where(self.core_pred_sub_labels[core] == self.diagnosis_dict[class_needed])
            chosen_amount_count = class_pred_locations[0].shape[0]
            
            # Same case as before, none predicated and core label is that core
            if self.core_true_label[core] == self.diagnosis_dict[class_needed] and chosen_amount_count == 0:
                highest_val_loc = np.argmax(self.core_probability_labels[core][:, self.diagnosis_dict[class_needed]])
                
                all_pred_mspec_vals[loc:loc+1] = self.core_spec[core][highest_val_loc]
                all_predicted_probability_values[loc:loc+1] = self.core_probability_labels[core][highest_val_loc][self.diagnosis_dict[class_needed]]
                loc+=1
            else:
                all_pred_mspec_vals[loc:loc+chosen_amount_count] = self.core_spec[core][class_pred_locations]
                all_predicted_probability_values[loc:loc+chosen_amount_count] = self.core_probability_labels[core][class_pred_locations, self.diagnosis_dict[class_needed]]
                
                loc += chosen_amount_count
           
        # Not enough needed
        if k >= all_predicted_probability_values.shape[0]:
            return (all_pred_mspec_vals)
        else:
            k_highest_val_locs =  np.argpartition(all_predicted_probability_values, -k)[-k:]
            k_highest_values = all_pred_mspec_vals[k_highest_val_locs]
            return(k_highest_values)
    
    def compute_params_all_cores(self, epoch, balance_training=False, balance_every_x = 5, train_non_healthy_only=True):
        
        total_tissue_count = {'high': 0, 'CA': 0, 'low': 0, 'healthy': 0} 
        epoch_cost = 0
        for core in self.smallROI.cores_list:
            
            if train_non_healthy_only:
                if self.core_true_label[core] == self.diagnosis_dict['healthy']:
                    core_cost, tissue_count = self._compute_params_single_core(core)
                else:
                    core_cost, tissue_count = self._compute_params_nonhealthy_only_single_core(core)
            else:
                core_cost, tissue_count = self._compute_params_single_core(core)
            
            total_tissue_count['high'] += tissue_count['high']
            total_tissue_count['CA'] += tissue_count['CA']
            total_tissue_count['low'] += tissue_count['low']
            total_tissue_count['healthy'] += tissue_count['healthy']
            epoch_cost+=core_cost
         
        print('Initial Pass Tissue Count: ', total_tissue_count)
        if balance_training and (epoch % balance_every_x == 0):
            additional_cost = self.balance_training(total_tissue_count)
            epoch_cost+=additional_cost
        return(epoch_cost)
            
    def _update_predicted_labels_single_core(self, core, two_class_per_core = True,balance_every_x=5):
        batch_idx = 0
        labels_changed = 0
        total_input_vals = self.core_probability_labels[core].shape[0]

        while (batch_idx+self.batch_size < total_input_vals):
            
            train_batch = self.core_spec[core][batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            previous_labels = self.core_pred_sub_labels[core][total_input_vals-self.batch_size:]

            preds = self.net1.single_core_predict_labels(train_batch, keep_prob=self.keep_prob)
            new_imputed_labels = one_hot_probability_to_single_label(preds, self.num_classes)
            
            if two_class_per_core:
                new_imputed_labels[np.where(new_imputed_labels!=self.diagnosis_dict['healthy'])] = self.core_true_label[core]
            self.core_probability_labels[core][total_input_vals-self.batch_size:] = preds
            
            diffs = len(np.where(previous_labels!=new_imputed_labels)[0])
            self.core_pred_sub_labels[core][batch_idx:batch_idx+self.batch_size] = new_imputed_labels
            labels_changed+=diffs
            
            batch_idx += self.batch_size
        
        train_batch = self.core_spec[core][total_input_vals-self.batch_size:, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        train_labels = self.core_pred_sub_labels[core][total_input_vals-self.batch_size:]
        train_labels = single_to_one_hot(train_labels, self.num_classes)
        
        preds = self.net1.single_core_predict_labels(train_batch, keep_prob=self.keep_prob)
        new_imputed_labels = one_hot_probability_to_single_label(preds, self.num_classes)
        if two_class_per_core:
            new_imputed_labels[np.where(new_imputed_labels!=self.diagnosis_dict['healthy'])] = self.core_true_label[core]
        self.core_probability_labels[core][total_input_vals-self.batch_size:] = preds
        
        diffs = len(np.where(previous_labels!=new_imputed_labels)[0])
        self.core_pred_sub_labels[core][total_input_vals-self.batch_size:] = new_imputed_labels
        labels_changed+=diffs
        
        return(labels_changed)
    
    def impute_labels_all_cores(self,two_class_per_core=False):
                
        labels_changed = 0
        for core in self.smallROI.cores_list:
            # dont impute healthy cores
            if self.core_true_label[core] == self.diagnosis_dict['healthy']:
                continue
            core_labels_changed = self._update_predicted_labels_single_core(core,two_class_per_core)
            labels_changed+=core_labels_changed
        return(labels_changed)
    
    def cnn_X_epoch(self, x, balance_classes=False, reset_healthy_count=False,balance_every_x=5,test_every_x=5, two_class_per_core=False, train_non_healthy_only=False, enforce_healthy_constraint=False):
        for epoch in range(1, x):
            print("Epoch ", epoch)
            
            cost = self.compute_params_all_cores(epoch, balance_training=balance_classes, balance_every_x=balance_every_x,train_non_healthy_only=train_non_healthy_only)
            print('    Cost: ', cost)            
            labels_changed = self.impute_labels_all_cores(two_class_per_core=two_class_per_core)
            print("    Labels Changed: ", labels_changed)
            self.enforce_label_constraints(enforce_healthy_constraint=enforce_healthy_constraint)
            if epoch % test_every_x == 0:
                self.eval_all_cores()
            if reset_healthy_count:
                self.reset_healthy_subtissue_input_count()
                
                
    # Impute the new labels and enforce label constraints
    def enforce_label_constraints(self, enforce_healthy_constraint=False):
        k_large_elements = self.batch_size
        for core in self.smallROI.cores_list:
            core_label = self.core_true_label[core]
            
            # Set all labels to healthy if core is labeled healthy
            if core_label == self.diagnosis_dict['healthy']:
                self.core_pred_sub_labels[core][:] =  self.diagnosis_dict['healthy']
            
            # Make sure at least one value is labeled non-healthy if core is non-healthy
            else:
                if np.sum(self.core_pred_sub_labels[core]) == 0 or np.where(self.core_pred_sub_labels[core] == core_label)[0].shape[0] == 0:
                    #Pick the max value specific to the core class
                    k_max_element_lcoations = np.argpartition(self.core_probability_labels[core][:, core_label], -k_large_elements)[-k_large_elements:]
                    self.core_pred_sub_labels[core][k_max_element_lcoations] = core_label
                    
                if enforce_healthy_constraint:
                    healthy_locations = np.where(self.core_pred_sub_labels[core] == self.diagnosis_dict['healthy'])
                    healthy_count = healthy_locations[0].shape[0]

                    if healthy_count <k_large_elements:
                        k_max_element_lcoations = np.argpartition(self.core_probability_labels[core][:, self.diagnosis_dict['healthy']], -k_large_elements)[-k_large_elements:]
                        self.core_pred_sub_labels[core][k_max_element_lcoations] = self.diagnosis_dict['healthy']
    
    # Evaulte according to only the given subtissue labels
    def eval_single_core_direct_labels(self, core):
        true_labels = self.core_true_sub_labels[core]
        pred_labels = self.core_pred_sub_labels[core]
        core_label = self.core_true_label[core]
        
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
    
    
    def eval_all_cores(self):
        # THis is after enforcing constraints
        #self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 0}
        tp_high = 0
        tp_CA = 0
        tp_low = 0
        tn_all_cores = 0
        
        total_high = 0
        total_CA = 0
        total_low = 0
        total_neg_all_cores = 0
        for core in self.smallROI.cores_list:
            if self.core_true_label[core] == self.diagnosis_dict['healthy']:
                continue
            tp, tn, total_pos, total_neg, core_label = self.eval_single_core_direct_labels(core)
            
            if core_label == self.diagnosis_dict['healthy']:
                tn_all_cores += tn
                total_neg_all_cores += total_neg
                
            elif core_label == self.diagnosis_dict['high']:
                tp_high += tp
                total_high += total_pos
                tn_all_cores += tn
                total_neg_all_cores += total_neg
                
            elif core_label == self.diagnosis_dict['CA']:
                tp_CA += tp
                total_CA += total_pos
                tn_all_cores += tn
                total_neg_all_cores += total_neg
                
            elif core_label == self.diagnosis_dict['low']:
                tp_low += tp
                total_low += total_pos
                tn_all_cores += tn
                total_neg_all_cores += total_neg
                
        print('  - Evaluating All Cores - ')
        accuracy = (tp_high + tp_CA + tp_low + tn_all_cores) / (total_high + total_CA + total_low +total_neg_all_cores)
        print('Accuracy: ', accuracy)
        
        if total_neg_all_cores == 0:
            neg_accuracy = -1
        else:
            neg_accuracy = (tn_all_cores / total_neg_all_cores)
        if total_high == 0:
            high_accuracy = -1
        else:
            high_accuracy = (tp_high / total_high)
        if total_CA == 0:
            ca_accuracy = -1
        else:
            ca_accuracy = (tp_CA / total_CA)
        if total_low == 0:
            low_accuracy = -1
        else:
            low_accuracy = (tp_low / total_low)
        
        print('Neg Accuracy: ', neg_accuracy)
        print('High Accuracy: ', high_accuracy)
        print('CA Accuracy: ', ca_accuracy)
        print('Low Accuracy: ', low_accuracy)
        
        balanced_accuracy = (neg_accuracy  * .25) + (high_accuracy*.25) + (ca_accuracy*.25) + (low_accuracy*.25)
        print('Balanced Accuracy: ', balanced_accuracy)
        print('  - Resuming Training  - ')
   
    def viz_single_core_pred(self, core):
        positions = self.positions[core]
        
        xmax = np.max(positions[:, 0])
        xmin = np.min(positions[:, 0])
        ymax = np.max(positions[:, 1])
        ymin = np.min(positions[:, 1])
        
        print('Xmax: ', xmax)
        print('Xmin: ', xmin)
        print('Ymax: ', ymax)
        print('Ymin: ', ymin)
        
        image_array = np.zeros((xmax-xmin+1, ymax-ymin+1))
        image_array[:] = 4
        print(image_array.shape)
        print("Core Label: ",  self.core_true_label[core])
        
        cmap = matplotlib.colors.ListedColormap(['white', 'red', 'blue', 'yellow', 'black'])
        bounds = [0, 1, 2, 3, 4,5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        for location in range(0, self.core_pred_sub_labels[core].shape[0]):
            label = self.core_pred_sub_labels[core][location]
            xloc = self.positions[core][location][0]- xmin
            yloc = self.positions[core][location][1] - ymin
            
            image_array[xloc, yloc] = label
        
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.grid(True)
        plt.imshow(image_array, interpolation='nearest',cmap = cmap,norm=norm)

        title = "Core Number: " + str(core)  + " Label: " +  self.diagnosis_dict_reverse[self.core_true_label[core]]
        print(title)
        plt.title(title)
        plt.colorbar(cmap=cmap,norm=norm,boundaries=bounds,ticks=[0,1,2, 3])
        filename = 'Images/Pred'+ str(int(core)) + '.png'
        print(filename)
        
        plt.savefig(filename, pad_inches=0)
        plt.clf()
    
    def viz_single_core_pred(self):
        for core in self.smallROI.cores_list:
            print(core)
            self.viz_single_core_true(core)
    

batch_size = 4
balance_classes=True # train on same amount of eachk class per epoch
balance_every_x = 2
two_class_per_core=True # make labels in each core only be healthy or the core label
reset_healthy_count = False # include healthy subtissue in non-healthy cores to count used for balancing classes
train_non_healthy_only = False # train on only the non-healthy assigned locations in non-healthy tissues
enforce_healthy_constraint = True
small_train = False

test_every_x = 1
num_epochs=150
lr=.001

#MIL = MIL(fc_units = 100, num_classes=4, width1=38,  width2=18, width3=16, filters_layer1=20, filters_layer2=40, filters_layer3=80, batch_size=batch_size, lr=lr, keep_prob=.99,small_train=small_train)
#MIL.init_MSI()
#MIL.cnn_X_epoch(num_epochs,balance_classes=balance_classes,reset_healthy_count=reset_healthy_count, balance_every_x=balance_every_x, test_every_x=test_every_x,two_class_per_core=two_class_per_core, train_non_healthy_only=train_non_healthy_only,enforce_healthy_constraint=enforce_healthy_constraint)

