#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:39:20 2021

@author: stephenschmidt
"""


from load_data import SmallROI
from load_train_data import H5MSI_Train
import numpy as np

def single_to_one_hot(labels, num_classes):
        #diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        # shifted to {'high': 0, 'CA': 1, 'low': 2, 'healthy': 3}
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for hot_class in range(0, num_classes):
        class_locations = np.where(labels == hot_class)
        one_hot_labels[class_locations, hot_class] = 1
    return(one_hot_labels)



class DirectCNNDataset():
    
    def __init__(self, num_classes=4):
        
        self.train_data_path = os.path.join(os.path.dirname(os.getcwd()), '89AccuracyOut')
        #self.train_data_path = os.path.join(os.path.dirname(os.getcwd()), 'MICNN_Out')
        
        self.train_data = H5MSI_Train()
        self.test_data = SmallROI()
        self.test_data.split_cores()
        
        self.num_classes=num_classes
        
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
        
        for core in self.train_data.cores_list:
   
            if 'Label' not in core:
                
                label_filename = core + "_multiclass.hdf5"
                label_file_path = os.path.join(self.train_data_path, label_filename)
                with h5py.File(label_file_path, "r") as hf:
                    dname = list(hf.keys())[0]
                    n1 = hf.get(dname)    
                    n1_array = np.copy(n1)
                    
                    one_hot_labels_array = single_to_one_hot(n1_array, 4)
                
                self.train_core_spec[core] = self.train_data.train_data[core]
                self.train_core_true_label[core] = int(self.train_data.train_data[core +'_Labels'][0])
                self.train_core_pred_sub_labels[core] = one_hot_labels_array.astype(int)
                self.train_core_probability_labels[core] = np.zeros((self.train_data.train_data[core].shape[0], self.num_classes))
                
        
        for core in self.test_data.cores_list:
            core_positions = self.test_data.core_specific_positions[core]
            self.test_core_spec[core] = self.test_data.spec[core_positions]
            self.test_core_true_sub_labels[core] = self.test_data.subtissue_labels[core_positions]
            self.test_core_true_label[core] = int(self.test_data.tissue_labels[core_positions][0])
            self.test_core_pred_sub_labels[core] = self.test_data.tissue_labels[core_positions].astype(int)
            self.test_core_probability_labels[core] = np.zeros((self.test_data.tissue_labels[core_positions].shape[0], self.num_classes))
            self.test_positions[core] = self.test_data.position[core_positions].astype(int)
            
            

        
        
        
        


dset = DirectCNNDataset(num_classes=4)