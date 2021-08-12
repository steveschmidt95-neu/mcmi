#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:39:20 2021

@author: stephenschmidt
"""


from load_data import SmallROI
from load_train_data import H5MSI_Train
import numpy as np
import os
import h5py

def single_to_one_hot(labels, num_classes):
        #diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        # shifted to {'high': 0, 'CA': 1, 'low': 2, 'healthy': 3}
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for hot_class in range(0, num_classes):
        class_locations = np.where(labels == hot_class)
        one_hot_labels[class_locations, hot_class] = 1
    return(one_hot_labels)

def get_cores_list_from_filename(filename):
    cores_list = []
    for h5_file in os.listdir(filename):
        cores_list.append(h5_file[0:-16])
    return(cores_list)


class DirectCNNDataset():
    
    def __init__(self, num_classes=4, spec_location = 'Data', train_data_location = 'MICNN_Out'):
        
        self.spec_path = os.path.join(os.path.dirname(os.getcwd()), spec_location)
        self.train_data_path = os.path.join(os.path.dirname(os.getcwd()), train_data_location)
        
        self.train_data = H5MSI_Train()
        
        
        for core in self.train_data.cores_list:
            spect_shape = self.train_data.train_data[core].shape[1]
            break
                
        cores_list = get_cores_list_from_filename(self.train_data_path)
        self.train_data.cores_list = cores_list
        
        print("Cores Included For Training: ", cores_list)
        
        self.num_classes=num_classes
        
        class_label_count = {}
        for i in range(num_classes):
            class_label_count[i] = 0
        
        for core in self.train_data.cores_list:
            if 'Label' not in core: 
                label_filename = core + "_multiclass.hdf5"
                label_file_path = os.path.join(self.train_data_path, label_filename)
                
                try:
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
                except FileNotFoundError:
                    print("Cant Find Core: ", core)
        
        self.spec_dict = {}
        index_dict = {}
        for i in range(num_classes):
            self.spec_dict[i] = np.zeros((class_label_count[i], spect_shape))
            index_dict[i] = 0  
        
        
        # put each labeled location into its own array so it can be flat
        for core in self.train_data.cores_list:
            if 'Label' not in core: 
                
                label_filename = core + "_multiclass.hdf5"
                label_file_path = os.path.join(self.train_data_path, label_filename)
                spec_path = os.path.join(self.spec_path, core + '.hdf5')
                
                try:
                    with h5py.File(label_file_path, "r") as hf:
                        with h5py.File(spec_path, "r") as hf_spec:
                            dname = list(hf.keys())[0]
                            n1 = hf.get(dname)    
                            label_array = np.copy(n1)
                            
                            dname = list(hf_spec.keys())[0]
                            n1 = hf_spec.get(dname)    
                            spec_array = np.copy(n1)
                                    
                            label_locations = np.where(label_array != 0)
                            count = label_locations[0].shape[0]
                            if count == 0:
                                label = 0
                                count = label_array.shape[0]
                            else:
                                label = int(label_array[label_locations][0])
                            
                            if label == 0:
                                self.spec_dict[label][index_dict[label]:index_dict[label]+count,:] = spec_array
                            else:
                                spec_wanted = np.take(spec_array, label_locations, 0)
                                self.spec_dict[label][index_dict[label]:index_dict[label]+count,:] = spec_wanted
                            index_dict[label] += count     
                except FileNotFoundError:
                    print("Cant Find Core: ", core)
                    
                    
    