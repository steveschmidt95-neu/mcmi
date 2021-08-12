#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 18:19:58 2021

@author: stephenschmidt
"""


import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1
v1.disable_eager_execution()
import time
import os
from net1_MIL import MSInet1
import matplotlib.pyplot as plt
import matplotlib
import h5py
import random



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


class SimData():
    
    def __init__(self, h5_name = 'sim_multiClass.h5'):
        self.data_folder = os.path.join(os.path.dirname(os.getcwd()), 'OriginalData')
        self.data_path = os.path.join(self.data_folder, h5_name)
        self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 0}
        self.tumor_dict = {'Stroma': 0, 'Tumor': 1}
        
        f = h5py.File(self.data_path, 'r')
        keys = list(f.keys())        
        dset = f['msidata']
        
        self.position = dset['position']
        self.spec = dset['spec']
        subtissue_labels = dset['subtissue_label']
        tissue_label = dset['tissue_label']
        sample = dset['sample']
        
        sample_number = np.zeros((sample.shape[0]))
        for row in range(0, sample.shape[0]):
            label = sample[row].decode('UTF-8')
            label = int(label[6:])
            sample_number[row] = label
        self.core = sample_number
        
        sub_labels = np.zeros((subtissue_labels.shape[0]))
        for row in range(0, subtissue_labels.shape[0]):
            label = subtissue_labels[row].decode('UTF-8')
            sub_labels[row] = self.diagnosis_dict[label]
        self.subtissue_labels = sub_labels
        
        tissue_labels_numbers = np.zeros((tissue_label.shape[0]))
        for row in range(0, tissue_label.shape[0]):
            label = tissue_label[row].decode('UTF-8')
            tissue_labels_numbers[row] = self.diagnosis_dict[label]
        self.tissue_labels = tissue_labels_numbers
        
        positions = np.zeros((self.position.shape[0], 2))
        for row in range(0, positions.shape[0]):
            positions[row, 0] = self.position[row][0]
            positions[row, 1] = self.position[row][1]
        self.position = positions
        
    def split_cores(self):
        self.cores_list = np.unique(self.core)
        self.core_specific_positions = {}
        
        for core in self.cores_list:            
            core_positions = np.where(self.core==core)            
            self.core_specific_positions[core] = core_positions
            
            
def get_cores_list_from_filename(filename):
    cores_list = []
    for h5_file in os.listdir(filename):
        cores_list.append(float(h5_file[0:-16]))
    return(cores_list)
            
class DirectCNNSimDataset():
    
    def __init__(self, num_classes=4, train_data_location = 'MICNNSim_Out'):
        
        self.train_data_path = os.path.join(os.path.dirname(os.getcwd()), train_data_location)
        
        self.train_data = SimData()
        self.train_data.split_cores()
        
        cores_list = get_cores_list_from_filename(self.train_data_path)
        self.train_data.cores_list = cores_list
        
        print("Cores Included For Training: ", cores_list)
        
        self.sim_core_spec = {}
        self.sim_core_true_sub_labels = {}
        self.sim_core_true_label = {}
        
        
        # Get the number of features shape for the sim input
        for core in self.train_data.cores_list:
            core_positions = self.train_data.core_specific_positions[core]
            self.spec_shape = self.train_data.spec[core_positions].shape[1]
            break
        
        for core in self.train_data.cores_list:
            core_positions = self.train_data.core_specific_positions[core]
            self.sim_core_spec[core] = self.train_data.spec[core_positions]
            self.sim_core_true_sub_labels[core] = self.train_data.subtissue_labels[core_positions]
            self.sim_core_true_label[core] = int(self.train_data.tissue_labels[core_positions][0])
        
        self.num_classes=num_classes
        
        class_label_count = {}
        for i in range(num_classes):
            class_label_count[i] = 0
        
        for core in self.train_data.cores_list:
            label_filename = str(core) + "_multiclass.hdf5"
            label_file_path = os.path.join(self.train_data_path, label_filename)
            
            with h5py.File(label_file_path, "r") as hf:
                dname = list(hf.keys())[0]
                n1 = hf.get(dname)    
                n1_array = np.copy(n1)
                
                label_locations = np.where(n1_array != 0)[0]
                count = label_locations.shape[0]
                if count == 0:
                    label = 0
                    count = n1_array.shape[0]
                else:
                
                    label = int(n1_array[label_locations][0])
                
                class_label_count[label] += count
        
        self.spec_dict = {}
        index_dict = {}
        for i in range(num_classes):
            self.spec_dict[i] = np.zeros((class_label_count[i], self.spec_shape))
            index_dict[i] = 0  
        
        
        
        # put each labeled location into its own array so it can be flat
        
        for core in self.train_data.cores_list:
            label_filename = str(core) + "_multiclass.hdf5"
            label_file_path = os.path.join(self.train_data_path, label_filename)
            
            spec_array = self.sim_core_spec[core]
            with h5py.File(label_file_path, "r") as hf:
                dname = list(hf.keys())[0]
                n1 = hf.get(dname)    
                label_array = np.copy(n1)

                label_locations = np.where(label_array != 0)
                count = label_locations[0].shape[0]
                if count == 0:
                    label = 0
                    count = label_array.shape[0]
                else:
                    label = int(label_array[label_locations][0])
           
                if label == 0:
                    self.spec_dict[label][index_dict[label]:index_dict[label]+count, :] = spec_array
                else:
                    spec_wanted = np.take(spec_array, label_locations, 0)
                    self.spec_dict[label][index_dict[label]:index_dict[label]+count, :] = spec_wanted
                index_dict[label] += count 
                        




