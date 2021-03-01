#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 19:24:51 2021

@author: stephenschmidt
"""

import os
import sys
import h5py
import numpy as np
import collections
import matplotlib.pyplot as plt

class ROI_Data():
    
    def __init__(self, msi_data, sliced_roi_positions):
        
        self.roi_num = int(sliced_roi_positions[0, 3])
        self.data = None
        self.data_folder = os.path.join(os.path.dirname(os.getcwd()), 'Data')
        
        self.xmin = int(np.min(sliced_roi_positions[:,0]))
        self.xmax = int(np.max(sliced_roi_positions[:,0]))
        self.ymin = int(np.min(sliced_roi_positions[:,1]))
        self.ymax = int(np.max(sliced_roi_positions[:,1]))            
        
        print("Ymax: ", self.ymax)
        print("Ymin: ", self.ymin)
        print("XMax: ", self.xmax)
        print("XMin: ", self.xmin)

        self.data = np.zeros((sliced_roi_positions.shape[0], msi_data.shape[1]))
        self.labels = np.zeros((sliced_roi_positions.shape[0]))
        
        for line in range(0, sliced_roi_positions.shape[0]):
            
            self.data[line, :] = msi_data[sliced_roi_positions[line, 5], :]
            self.labels[line] = sliced_roi_positions[line, 4]
        
        
    def save_h5(self):
        
        data_filename = os.path.join(self.data_folder, 'ROI' + str(self.roi_num)+ '.hdf5')
        assert (len(self.data.shape)==2)
        with h5py.File(data_filename, "w") as f:
            dset = f.create_dataset("roi" + str(self.roi_num) + "data", data=self.data, dtype='f')
            
        labels_filename = os.path.join(self.data_folder, 'ROI'+ str(self.roi_num) + 'Labels' + '.hdf5')
        with h5py.File(labels_filename, "w") as f:
            dset = f.create_dataset("roi" + str(self.roi_num) + "labels", data=self.labels, dtype='f')


class MSIData():
    
    def __init__(self, data_name = 'msi.h5', tab_name = 'final_annotation_complete_cores_2.tabular'):
        
        self.data_folder = os.path.join(os.path.dirname(os.getcwd()), 'OriginalData')
        self.data_path = os.path.join(self.data_folder, data_name)
        self.tab_path = os.path.join(self.data_folder, tab_name)
        self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        
        f = h5py.File(self.data_path, 'r')
        keys = list(f.keys())
        print("H5 Keys: ", keys)
        dset = f['spec']
        
        self.msispec = dset['msispec']
        self.position = dset['position']
        
        print('MSI Data Shape: ', self.msispec.shape)
        print('Position Shape: ', self.position.shape)
        
        line_count = 0
        for line in open(self.tab_path):
            line_data = line.split()
            line_count += 1
            
        position_data = np.zeros((line_count-1, 6))
        
        line_count = 0
        for line in open(self.tab_path):
            if line_count == 0:
                line_count+=1
                continue
            line_data = line.split() 
            position_data[line_count-1, 0] = int(line_data[0]) # X
            position_data[line_count-1, 1] = int(line_data[1]) # Y
            position_data[line_count-1, 2] = int(line_data[2][-1]) # TMA?
            position_data[line_count-1, 3] = int(line_data[3][3:]) # ROINUM
            
            if position_data[line_count-1, 2] == 2:  # if TMA == 2, shift up by 100
                position_data[line_count-1, 3] += 100
            position_data[line_count-1, 4] = self.diagnosis_dict[line_data[4]] # change diagnosis to numercial
            position_data[line_count-1, 5] = line_count - 2
            
            line_count+=1
            
        self.position_data = position_data
        
        print('Number of ROIs', np.max(position_data[:, 3]))
        num_rois = int(np.max(position_data[:, 3]))
        
        for roi in range(1, num_rois):
            print('-'*20)
            print('ROI NUM: ', roi)
            
            sliced_positions = position_data[np.where(position_data[:,3] == roi)]
            
            if sliced_positions.shape[0] == 0:
                print("No ROI values")
                continue
            
            roi_data = ROI_Data(self.msispec, sliced_positions)
            roi_data.save_h5()    
            
class H5MSI():
    
    def __init__(self):
        self.data_folder = os.path.join(os.path.dirname(os.getcwd()), 'Data')
        self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        
        h5_files = os.listdir(self.data_folder)
        h5_files = [elem for elem in h5_files if elem.endswith('.hdf5')]
        self.num_rois = len(h5_files)/2
        self.data_files = {}
        self.train_files = {}
        self.val_files = {}
        self.num_classes = 4
        
        for h5_file in h5_files:
            roi_path = os.path.join(self.data_folder, h5_file)
            
            if 'Labels' in roi_path:
                idx = h5_file.find('.')
                dict_key = h5_file[0:idx]
            else:
                idx = h5_file.find('.')
                dict_key = h5_file[0:idx]

            with h5py.File(roi_path, "r") as hf:
                dname = list(hf.keys())[0]
                n1 = hf.get(dname)    
                n1_array = np.copy(n1)
                
                self.data_files[dict_key] = n1_array
        
        print('Number of Samples in h5 datset: ', len(self.data_files.keys())//2)
        
        for key in self.data_files.keys():
            if not ('Labels' in key) and (len(key) > 5):
                self.val_files[key] = self.data_files[key]
                self.val_files[key+ 'Labels'] = self.data_files[key + 'Labels']
                
            if not ('Labels' in key) and ( len(key) <= 5):
                self.train_files[key] = self.data_files[key]
                self.train_files[key + 'Labels'] = self.data_files[key + 'Labels']
                
    def histo_data(self, val = 0):
        histo_count = np.zeros((5))
        x_axis = [1, 2, 3, 4]
        
        if val == 1:
            for key in list(self.val_files.keys()):
                if 'Label' in key:
                    array = self.val_files[key]
                    occurances = collections.Counter(array)
                    loc = int(list(occurances.keys())[0])
                    histo_count[loc] += occurances[loc]
                    plt.title('Validation Data')
        elif val ==0:
            for key in list(self.train_files.keys()):
                if 'Label' in key:
                    array = self.train_files[key]
                    occurances = collections.Counter(array)
                    loc = int(list(occurances.keys())[0])
                    histo_count[loc] += occurances[loc]
                    plt.title('Training Data')
                
        plt.bar(x_axis, histo_count[1:])
        
        plt.show()
        
        histo_count = np.zeros((5))
        x_axis = [1, 2, 3, 4]
        
    def two_class_data(self):
        self.num_classes=2
        for key in list(self.val_files.keys()):
                if 'Label' in key:
                    array = self.val_files[key]
                    cancer_class_locs = np.where(array!=4)
                    array[cancer_class_locs] = 0
                    array[array!=0] = 1
                    self.val_files[key] = array
                    
                    
                    
        for key in list(self.train_files.keys()):
                if 'Label' in key:
                    array = self.train_files[key]
                    cancer_class_locs = np.where(array!=4)
                    array[cancer_class_locs] = 0
                    array[array!=0] = 1
                    self.train_files[key] = array
                    
    def flatten_data(self):
        
        total_train_length = 0
        for key in list(self.train_files.keys()):
            if 'Label' not in key:
                array = self.train_files[key]
                input_shape = self.train_files[key].shape[1]
                total_train_length += array.shape[0]

        train_loc = 0
        flat_train = np.zeros((total_train_length,input_shape))
        flat_train_labels = np.zeros((total_train_length))

        for key in list(self.train_files.keys()):
            if 'Label' not in key:
                array = self.train_files[key]
                
                assert (len(array.shape)==2)
                
                flat_train[train_loc:(array.shape[0]+train_loc), :] = array
                
                label_key = key + 'Labels'
                labels_array = self.train_files[label_key]
                
                assert labels_array.shape[0] == array.shape[0]
                
                flat_train_labels[train_loc:(array.shape[0])+train_loc] = labels_array
                train_loc += array.shape[0]
                
        self.flat_train = flat_train
        
        
        total_train_length = 0
        for key in list(self.val_files.keys()):
            if 'Label' not in key:
                array = self.val_files[key]
                input_shape = self.val_files[key].shape[1]
                total_train_length += array.shape[0]

        val_loc = 0
        flat_val = np.zeros((total_train_length,input_shape))
        flat_val_labels = np.zeros((total_train_length))

        for key in list(self.val_files.keys()):
            if 'Label' not in key:
                array = self.val_files[key]
                assert (len(array.shape)==2)
                flat_val[val_loc:(array.shape[0]+val_loc), :] = array
                
                label_key = key + 'Labels'
                labels_array = self.val_files[label_key]
                
                assert labels_array.shape[0] == array.shape[0]
                
                flat_val_labels[val_loc:(array.shape[0])+val_loc] = labels_array
                val_loc += array.shape[0]
        self.flat_val = flat_val
        
        #flat_val_labels   = flat_val_labels - 1
        #flat_train_labels = flat_train_labels - 1
        #self.flat_val_labels = flat_val_labels
        #self.flat_train_labels = flat_train_labels

        
        # Convert to 1 hot labels        
        new_flat_train_labels = np.zeros((flat_train_labels.shape[0], self.num_classes))
        if self.num_classes == 4:    
            for idx in range(0, flat_train_labels.shape[0]):
                new_flat_train_labels[idx, int(flat_train_labels[idx])-1] = 1
                
        elif self.num_classes == 2:
            for idx in range(0, flat_train_labels.shape[0]):
                new_flat_train_labels[idx, int(flat_train_labels[idx])] = 1
        
        else:
            assert False # need 2 or 4 classes
            
        new_flat_val_labels = np.zeros((flat_val_labels.shape[0], self.num_classes))
        if self.num_classes == 4:    
            for idx in range(0, flat_val_labels.shape[0]):
                new_flat_val_labels[idx, int(flat_val_labels[idx])-1] = 1
                
        elif self.num_classes == 2:
            for idx in range(0, flat_val_labels.shape[0]):
                new_flat_val_labels[idx, int(flat_val_labels[idx])] = 1
        
        else:
            assert False # need 2 or 4 classes
            
        self.flat_val_labels = new_flat_val_labels
        self.flat_train_labels = new_flat_train_labels
        
        
        
#redo_msi = MSIData()

#msi = H5MSI()
#msi.histo_data(val=0)
#msi.two_class_data()
#msi.flatten_data()

        