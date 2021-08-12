#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:16:10 2021

@author: stephenschmidt
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
import h5py


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
    if (uncertain_pred_locations[0].shape[0]>0):
        argmax_labels[uncertain_pred_locations[0]] = np.argmax(prev_labels[uncertain_pred_locations])
    return(argmax_labels)


class MIL():
    
    def __init__(self, fc_units = 50, num_classes=4, width1=38, width2=18, width3=16, filters_layer1=12, 
                 filters_layer2=24, filters_layer3=48, batch_size=4, lr=.001, keep_prob=.99, undertrain=True):
        
        self.train_data_total = H5MSI_Train()
        self.train_data_total.one_hot_labels()
            
            
        self.out_path = os.path.join(os.path.dirname(os.getcwd()), 'MICNN_Out')
        
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
        
        self.init_MSI()
        
        if (len(self.train_data_total.cores_list) == 0):
            print("No Cores Found")
            assert False # No Cores Found
        
        print("Cores Included for Training: ", self.train_data_total.cores_list)
    
        self.sample_shape = int(self.train_core_spec[self.train_data_total.cores_list[0]].shape[1])
        self.net1 = MSInet1(data_shape = self.sample_shape, fc_units=fc_units, num_classes=num_classes, width1=width1, 
                            width2=width2, width3=width3, filters_layer1=filters_layer1, filters_layer2=filters_layer2, 
                            filters_layer3=filters_layer3, batch_size=batch_size,lr=lr)
        self.net1.build_graph()
        
        #For Use in undertraining
        self.limiting_factor = 0
        self.positive_predicted = 0
        self.negative_predicted = 0
        self.positive_trained = 0
        self.negative_trained = 0
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
            
                        
        for core in self.train_data_total.cores_list:
            if 'Label' not in core:
                self.train_core_spec[core] = self.train_data_total.train_data[core]
                self.train_core_true_label[core] = int(self.train_data_total.train_data[core +'_Labels'][0])
                self.train_core_pred_sub_labels[core] = self.train_data_total.flat_labels[core +'_Labels'].astype(int)
                self.train_core_probability_labels[core] = np.zeros((self.train_data_total.train_data[core].shape[0], self.num_classes))
                #print("Core: ", core, " Label: ", self.train_core_true_label[core], "Shape: ", self.train_core_pred_sub_labels[core].shape)

        self.test_total_healthy_subtissue = self.count_healthy_locs_core_only()
    
    def save_all_cores_multiLabel(self):
        print("Saving Cores . . .")
        for core in self.train_data_total.cores_list:
            true_multiclass_label = self.train_core_true_label[core]
            
            # healthy class just save all 0's
            prev_labels = self.train_core_pred_sub_labels[core]
            if true_multiclass_label == 0:
                prev_labels = np.zeros((prev_labels.shape[0]))
            
            # the other labels are goin to be only 1's so its save to multiply them by their true core label
            else:
                prev_labels = np.argmax(prev_labels, axis=1)
                prev_labels *= true_multiclass_label
            
            labels_filename = os.path.join(self.out_path, core + '_multiclass.hdf5')
            with h5py.File(labels_filename, "w") as f:
                dset = f.create_dataset(core + "multiclass_labels", data=prev_labels, dtype='f')
    
    def _compute_params_healthy_only_single_core2class_hetero(self, core):
        
        total_cost = 0
        batch_idx  = 0
        cost = 0
        
        if(self.limiting_factor==1 and (self.negative_trained >= self.positive_predicted)):
            self.neg_train_limit_hit = 1
            return(total_cost)
        
        valid_input_val_locs = np.where(self.train_core_pred_sub_labels[core][:, 0]==1)[0]
        valid_input_val_locs = list(valid_input_val_locs)
        np.random.shuffle(valid_input_val_locs)
        total_input_vals = len(valid_input_val_locs) 
        
        spec = np.copy(self.train_core_spec[core])
        spec = spec[valid_input_val_locs]
        
        while (batch_idx+self.batch_size < total_input_vals):
            train_batch = spec[batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            train_labels = np.zeros((self.batch_size, self.num_classes))
            train_labels[:, 0] = 1

            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            total_cost+=cost
            self.train_core_probability_labels[core][batch_idx:batch_idx+self.batch_size] = preds
            
            # Hit the limit on training healthy subtissues
            self.negative_trained += batch_size
            if(self.limiting_factor==1 and (self.negative_trained >= self.positive_predicted)):
                self.neg_train_limit_hit = 1
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
            self.negative_trained += batch_size
        except ValueError:
            print("Not Enough Healthy Tissues in Hetero COre")
            print("Wring Shape: ", train_batch.shape)
        
        return(total_cost)
    
    def _compute_params_nonhealthy_only_single_core2class_hetero(self, core):
        
        total_cost = 0
        batch_idx  = 0
        cost = 0
        
        if(self.limiting_factor==0 and (self.positive_trained >= self.negative_predicted)):
            self.pos_train_limit_hit = 1
            return(total_cost)        
        
        valid_input_val_locs = np.where(self.train_core_pred_sub_labels[core][:, 1]==1)[0]
        valid_input_val_locs = list(valid_input_val_locs)
        np.random.shuffle(valid_input_val_locs)
        total_input_vals = len(valid_input_val_locs)
        
        spec = np.copy(self.train_core_spec[core])
        spec = spec[valid_input_val_locs]
        
        while (batch_idx+self.batch_size < total_input_vals):
            train_batch = spec[batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            train_labels = np.zeros((self.batch_size, self.num_classes))
            train_labels[:, 1] = 1

            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            total_cost+=cost
            self.train_core_probability_labels[core][batch_idx:batch_idx+self.batch_size] = preds
            
            # Hit the limit on training non-healthy subtissues
            self.positive_trained += batch_size
            if(self.limiting_factor==0 and (self.positive_trained >= self.negative_predicted)):
                self.pos_train_limit_hit = 1
                return(total_cost)
            
            batch_idx += self.batch_size
        
        if batch_idx == total_input_vals:
            return(total_cost)
            
        train_batch = spec[((self.batch_size*-1)):, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        train_labels = np.zeros((self.batch_size, self.num_classes))
        train_labels[:, 1] = 1
        
        try:
            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            self.train_core_probability_labels[core][(self.batch_size*-1):, :] = preds
            total_cost+=cost
            self.positive_trained += batch_size
        except ValueError:
            print("Wring Shape: ", train_batch.shape)
        
        return(total_cost)
    
    # Heterogenous classes training at the same time
    def _compute_params_single_core_2class(self, core):
        cost = 0
        
        total_cost = 0
        batch_idx  = 0
        total_input_vals = self.core_probability_labels[core].shape[0]
        
        while (batch_idx+self.batch_size < total_input_vals):
            
            train_batch = self.train_core_spec[core][batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            train_labels = self.train_core_pred_sub_labels[core][batch_idx:batch_idx+self.batch_size]

            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            total_cost+=cost
            self.core_probability_labels[core][batch_idx:batch_idx+self.batch_size] = preds
            batch_idx += self.batch_size
        
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
        
        total_cost = 0
        batch_idx  = 0
        
        if(self.limiting_factor==1 and (self.negative_trained >= self.positive_predicted)):
            self.neg_train_limit_hit = 1
            return(total_cost)
        
        total_input_vals = self.train_core_probability_labels[core].shape[0]
        
        spec_local = np.copy(self.train_core_spec[core])
        np.random.shuffle(spec_local)
        
        while (batch_idx+self.batch_size < total_input_vals):
            
            train_batch = spec_local[batch_idx:batch_idx+self.batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            train_labels = np.zeros((self.batch_size, self.num_classes))
            train_labels[:, 0] = 1
            
            cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
            total_cost+=cost
            self.train_core_probability_labels[core][batch_idx:batch_idx+self.batch_size] = preds
            batch_idx += self.batch_size
            
            self.negative_trained += batch_size
            if(self.limiting_factor==1 and (self.negative_trained >= self.positive_predicted)):
                self.neg_train_limit_hit = 1
                return(total_cost)
        
        if batch_idx == total_input_vals:
            return(total_cost)
        
        train_batch = spec_local[((self.batch_size)*-1):, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        train_labels = np.zeros((self.batch_size, self.num_classes))
        train_labels[:, 0] = 1
        
        cost, preds = self.net1.single_core_compute_params(train_batch, train_labels, keep_prob=self.keep_prob)
        self.train_core_probability_labels[core][(self.batch_size*-1):, :] = preds
        total_cost+=cost
        self.negative_trained += batch_size
        
        return(total_cost)
    
    def compute_params_all_cores2class(self, epoch, balance_training=False, balance_every_x = 5, train_non_healthy_only=True):
        
        #Reset the undertrain counters
        self.pos_train_limit_hit = 0
        self.neg_train_limit_hit = 0
        self.negative_trained = 0
        self.positive_trained = 0
        
        # Giving head start so i t won't perpetually train more of one class
        if self.positive_predicted > self.negative_predicted:
            pass
            
        
        epoch_cost = 0
        for core in self.train_data_total.cores_list:

            if self.undertrain:
                # If healthy, can train on all the labels in the core with this function
                if self.train_core_true_label[core] == self.diagnosis_dict['healthy']:
                    core_cost = self._compute_params_single_core_2class_healthy_homogenous(core)
                else:
                    # If not, train healthy and then noin-healthy seperatly counting each 
                    non_healthy_cost = self._compute_params_nonhealthy_only_single_core2class_hetero(core)
                    healthy_cost = self._compute_params_healthy_only_single_core2class_hetero(core)
                    core_cost = non_healthy_cost + healthy_cost
            else:
                core_cost = self._compute_params_single_core_2class(core)
            epoch_cost+=core_cost
        print("Healthy trained: ", self.negative_trained)
        print("Positive trained: ", self.positive_trained)

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
            new_imputed_labels = single_to_one_hot(new_imputed_labels, 2)
            self.train_core_probability_labels[core][total_input_vals-self.batch_size:] = preds
            
            diffs = len(np.where(previous_labels!=new_imputed_labels)[0])
            
            self.train_core_pred_sub_labels[core][batch_idx:batch_idx+self.batch_size] = new_imputed_labels
            
            labels_changed+= (diffs/2)
            
            batch_idx += self.batch_size

        train_batch = self.train_core_spec[core][total_input_vals-self.batch_size:, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        previous_labels = self.train_core_pred_sub_labels[core][total_input_vals-self.batch_size:]
    
        preds = self.net1.single_core_predict_labels(train_batch, keep_prob=self.keep_prob)
        new_imputed_labels = one_hot_probability_to_single_label(preds, previous_labels, self.num_classes)
        if two_class_per_core:
            new_imputed_labels[np.where(new_imputed_labels!=self.diagnosis_dict['healthy'])] = self.core_true_label[core]
        self.train_core_probability_labels[core][total_input_vals-self.batch_size:] = preds
        
        new_imputed_labels = single_to_one_hot(new_imputed_labels, 2)
        diffs = len(np.where(previous_labels!=new_imputed_labels)[0])
        self.train_core_pred_sub_labels[core][total_input_vals-self.batch_size:] = new_imputed_labels
        labels_changed+=diffs
        
        return(labels_changed)
        
    
    def impute_labels_all_cores(self,two_class_per_core=False):
        print("Imputing Labels")
        labels_changed = 0
        for core in self.train_data_total.cores_list:
            # dont impute healthy cores
            if not self.train_core_true_label[core] == self.diagnosis_dict['healthy']:
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

            positive_predicted = (len(np.where(self.train_core_pred_sub_labels[core][:,1] == 1)[0]))
            healthy_predicted = int(total_subtissues - positive_predicted)
            
            positive_predicted_total+=positive_predicted
            healthy_predicted_total+=healthy_predicted
            
        if positive_predicted_total < healthy_predicted_total:
            limiting_factor= 1
        
        print("Limiting Factor: ", limiting_factor)
        assert (healthy_predicted_total + positive_predicted_total) == total_count
        self.negative_predicted = healthy_predicted_total
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

            positive_predicted = (len(np.where(self.train_core_pred_sub_labels[core][:,1] == 1)[0]))
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
            
        self.save_all_cores_multiLabel()
                
                
    # Impute the new labels and enforce label constraints
    def enforce_label_constraints(self, enforce_healthy_constraint=False):
        k_large_elements = self.batch_size
        for core in self.train_data_total.cores_list:
            core_label = self.train_core_true_label[core]
            if core_label !=0:
                core_label = 1
            
            # Set all labels to healthy if core is labeled healthy
            if core_label == self.diagnosis_dict['healthy']:
                self.train_core_pred_sub_labels[core][:, 0] =  1
                self.train_core_pred_sub_labels[core][:, 1] =  0
            
            # Make sure at least k values is labeled non-healthy if core is non-healthy
            else:
                # All labels are healthy
                if np.sum(self.train_core_pred_sub_labels[core][:,1]) < k_large_elements:
                    #Pick the max values specific to the core class
                    k_max_element_lcoations = np.argpartition(self.train_core_probability_labels[core][:, 1], -k_large_elements)[-k_large_elements:]
                    self.train_core_pred_sub_labels[core][k_max_element_lcoations, 0] = 0
                    self.train_core_pred_sub_labels[core][k_max_element_lcoations, 1] = 1
                    
                    try:
                        assert np.sum(self.train_core_pred_sub_labels[core][:,1])  >= k_large_elements
                    
                    except:
                        print("Error with non-healthy")
                        print(k_max_element_lcoations)
                        print(np.sum(self.train_core_pred_sub_labels[core][:,1]))
                        self.train_core_pred_sub_labels[core]
                        assert False
                
                # apply the same rule to healthy tissue, not leaving a tissue with comptely unhealthy labels
                if enforce_healthy_constraint:
                    if np.sum(self.train_core_pred_sub_labels[core][:,0]) < k_large_elements:
                        #Pick the max values specific to the core class
                        k_max_element_lcoations = np.argpartition(self.train_core_probability_labels[core][:, 0], -k_large_elements)[-k_large_elements:]
                        self.train_core_pred_sub_labels[core][k_max_element_lcoations, 0] = 1
                        self.train_core_pred_sub_labels[core][k_max_element_lcoations, 1] = 0
                        
                        try:
                            assert np.sum(self.train_core_pred_sub_labels[core][:,0]) >= k_large_elements
                        
                        except:
                            print("Error with healthy")
                            print(k_max_element_lcoations)
                            print(np.sum(self.train_core_pred_sub_labels[core][:,0]))
                            print(self.train_core_pred_sub_labels[core])
                            assert False            
    
    
batch_size = 8
enforce_healthy_constraint = True # Enforce the same constraint for healthy tissus on non-healthy cores

test_every_x = 1
num_epochs=3
lr=.001

MIL = MIL(fc_units = 100, num_classes=2, width1=38,  width2=18, width3=16, filters_layer1=40, 
    filters_layer2=60, filters_layer3=100, batch_size=batch_size, lr=lr, keep_prob=.99)
MIL.init_MSI()
MIL.cnn_X_epoch(num_epochs, test_every_x=test_every_x,enforce_healthy_constraint=enforce_healthy_constraint)
