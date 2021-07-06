

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
    
    def __init__(self, msi_data, sliced_roi_positions, tuple_positions):
        
        self.roi_num = int(sliced_roi_positions[0, 5])
        self.data = None
        self.data_folder = os.path.join(os.path.dirname(os.getcwd()), 'Data')
        
        self.data = np.zeros((sliced_roi_positions.shape[0], msi_data.shape[1]))
        self.labels = np.zeros((sliced_roi_positions.shape[0], 2)) # first spot is core label, second is subtissue
        miss_count = 0
        miss_lines = []
        
        for line in range(0, sliced_roi_positions.shape[0]):
            print(self.roi_num, ': ' , line, ' / ', sliced_roi_positions.shape[0])
            x = int(sliced_roi_positions[line, 0])
            y = int(sliced_roi_positions[line, 1])
            
            
            found_msi_data = False
            for position_row in range(0, tuple_positions.shape[0]):
                position_tuple = tuple_positions[position_row]
                checkx = position_tuple[0]
                checky = position_tuple[1]
                
                if checkx == x and checky == y:
                    found_msi_data = True
                    self.data[line, :] = msi_data[position_row, :]
                    self.labels[line,0] = sliced_roi_positions[line, 3]
                    self.labels[line, 1] = sliced_roi_positions[line, 4]
                        
            if not found_msi_data:
                #assert False # no msi data found for this
                miss_count +=1
                miss_lines.append(line)
                
        print('Missing Values: ', miss_count)
        self.labels = np.delete(self.labels, miss_lines, 0)
        self.data = np.delete(self.data, miss_lines, 0)
        assert self.labels.shape[0] == self.data.shape[0]
        
        
    def save_h5(self):
        
        data_filename = os.path.join(self.data_folder, 'ROI' + str(self.roi_num)+ '.hdf5')
        assert (len(self.data.shape)==2)
        with h5py.File(data_filename, "w") as f:
            dset = f.create_dataset("roi" + str(self.roi_num) + "data", data=self.data, dtype='f')
            
        labels_filename = os.path.join(self.data_folder, 'ROI'+ str(self.roi_num) + 'Labels' + '.hdf5')
        with h5py.File(labels_filename, "w") as f:
            dset = f.create_dataset("roi" + str(self.roi_num) + "labels", data=self.labels, dtype='f')


class MSIData():
    
    def __init__(self, data_name = 'msi.h5', sub_tab_name = 'subtissue_labels.tabular'):
        
        self.data_folder = os.path.join(os.path.dirname(os.getcwd()), 'OriginalData')
        self.data_path = os.path.join(self.data_folder, data_name)
        self.sub_tab_path = os.path.join(self.data_folder, sub_tab_name)
        self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        self.tumor_dict = {'Stroma': 0, 'Tumor': 1}
        
        f = h5py.File(self.data_path, 'r')
        keys = list(f.keys())
        print("H5 Keys: ", keys)
        dset = f['spec']
        
        self.msispec = dset['msispec']
        self.position = dset['position']
        
                
        
        # put all entries into a matrix
        line_count = 0
        for line in open(self.sub_tab_path):
            line_data = line.split()
            line_count += 1
            
        position_data = np.zeros((line_count-1, 6))
        line_count = 0
        for line in open(self.sub_tab_path):
            if line_count == 0:
                line_count+=1
                continue
            line_data = line.split() 
            position_data[line_count-1, 0] = int(line_data[0]) # X
            position_data[line_count-1, 1] = int(line_data[1]) # Y
            position_data[line_count-1, 2] = int(line_data[2][-1]) # SlideNum
            position_data[line_count-1, 3] = self.tumor_dict[line_data[3]] # 0 for Stroma, 1 for tumor
            position_data[line_count-1, 4] = self.diagnosis_dict[line_data[4]]
            position_data[line_count-1, 5] = int(line_data[5][3:]) # ROINUM
            
            line_count+=1
        self.subtissue_position_data = position_data
        
        # put subtissue entries into a matrix
        line_count = 0
        for line in open(self.sub_tab_path):
            line_data = line.split()
            line_count += 1

        
        #build each ROI one at a time
        print('Number of ROIs', np.max(self.subtissue_position_data[:, 5]))
        num_rois = int(np.max(position_data[:, 5]))
        
        for roi in range(1, num_rois):
            print('-'*20)
            print('ROI NUM: ', roi)
            
            sliced_positions = self.subtissue_position_data[np.where(self.subtissue_position_data[:,5] == roi)]
            
            if sliced_positions.shape[0] == 0: # some ROI's dont exist
                print("No ROI values")
                continue
            
            roi_data = ROI_Data(self.msispec, sliced_positions, self.position)
            roi_data.save_h5()    

def single_to_one_hot(labels, num_classes):
        #diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        # shifted to {'high': 0, 'CA': 1, 'low': 2, 'healthy': 3}
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for hot_class in range(0, num_classes):
        class_locations = np.where(labels == hot_class)
        one_hot_labels[class_locations, hot_class] = 1
        
    return(one_hot_labels)

def one_hot_to_single(labels, num_classes):
    pass
    

# Shuffle data and labels together
def shuffle_data(data, labels, num_classes):
    if len(labels.shape) == 1: # need to convert to one hot
        labels = single_to_one_hot(labels, num_classes)
    
    train_with_labels = np.zeros((data.shape[0], data.shape[1] + num_classes))
    train_with_labels[:, 0:data.shape[1]] = data
    train_with_labels[:, data.shape[1]:data.shape[1]+num_classes]= labels[:]
    np.random.shuffle(train_with_labels)
    
    shuffled_data = train_with_labels[:, 0:data.shape[1]]
    shuffled_labels = train_with_labels[:, ((num_classes)*-1):]
      
    return(shuffled_data, shuffled_labels)       
    
class H5MSI():
    
    def __init__(self):
        self.data_folder = os.path.join(os.path.dirname(os.getcwd()), 'Data')
        self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        
        h5_files = os.listdir(self.data_folder)
        h5_files = [elem for elem in h5_files if elem.endswith('.hdf5')]
        self.num_rois = len(h5_files)/2
        self.data_files = {}
        self.train_files = {}
        self.num_classes = 4
        self.data_is_flat = False
        
        for h5_file in h5_files:
            roi_path = os.path.join(self.data_folder, h5_file)
            
 
            idx = h5_file.find('.')
            dict_key = h5_file[0:idx]

            with h5py.File(roi_path, "r") as hf:
                dname = list(hf.keys())[0]
                n1 = hf.get(dname)    
                n1_array = np.copy(n1)
                
                self.data_files[dict_key] = n1_array
        
        for key in self.data_files.keys():
            if not ('Labels' in key):
                self.train_files[key] = self.data_files[key]
                self.train_files[key + 'Labels'] = self.data_files[key + 'Labels']
        
    def two_class_data(self):
        if not self.data_is_flat:
            print('Data Needs to be flattened first')
            assert False
        self.num_classes=2                                                            
        self.flat_train_labels[np.where(self.flat_train_labels!=self.diagnosis_dict['healthy'])] = 1 # tumor = 1
        self.flat_train_labels[np.where(self.flat_train_labels==self.diagnosis_dict['healthy'])] = 0 # healthy = 0
                    
    def flatten_data(self):
        total_train_length = 0
        for key in list(self.train_files.keys()):
            if 'Label' not in key:
                array = self.train_files[key]
                input_shape = self.train_files[key].shape[1]
                total_train_length += array.shape[0]

        train_loc = 0
        flat_train = np.zeros((total_train_length,input_shape))
        flat_train_labels = np.zeros((total_train_length, 2))

        for key in list(self.train_files.keys()):
            if 'Label' not in key:
                array = self.train_files[key]
                assert (len(array.shape)==2)
                flat_train[train_loc:(array.shape[0]+train_loc), :] = array

                label_key = key + 'Labels'
                labels_array = self.train_files[label_key]
                
                assert labels_array.shape[0] == array.shape[0]
                flat_train_labels[train_loc:(array.shape[0])+train_loc, :] = labels_array
                train_loc += array.shape[0]
        
        # Change stroma lables to healthy
        stroma_locations = np.where(flat_train_labels[:,0]==0)
        flat_train_labels[stroma_locations,1] = self.diagnosis_dict['healthy']
        flat_train_labels = flat_train_labels[:,1]
                
        self.flat_train = flat_train
        self.flat_train_labels = flat_train_labels
        self.data_is_flat = True

        
    def one_hot_labels(self):
        flat_train_labels = self.flat_train_labels

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
        self.flat_train_labels = new_flat_train_labels
        
    # data is split into train, test, 70:30
    # shuffling beforehand
    def split_data(self, train_split_amount = .7):
        if not self.data_is_flat:
            print('Flatten First')
            assert False
        
        if self.num_classes ==2:
            shuffled_data, shuffled_labels = shuffle_data(self.flat_train, self.flat_train_labels, self.num_classes)
            train_split_end = int(shuffled_data.shape[0]*.7)
            self.split_train = shuffled_data[0:train_split_end, :]
            self.split_train_labels = shuffled_labels[0:train_split_end, :]
            self.split_val = shuffled_data[train_split_end:, :]
            self.split_val_labels = shuffled_labels[train_split_end:, :]
        else:
            assert False # havent set this up for multi class yet
        
class SmallROI():
    
    def __init__(self, h5_name = 'smallROI.h5', num_classes= 4):
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
        core = dset['core']
        
        core_number = np.zeros((core.shape[0]))
        for row in range(0, core.shape[0]):
            label = core[row].decode('UTF-8')
            label = int(label[3:])
            core_number[row] = label
        self.core = core_number
        
        sub_labels = np.zeros((subtissue_labels.shape[0]))
        for row in range(0, subtissue_labels.shape[0]):
            label = subtissue_labels[row].decode('UTF-8')
            if num_classes == 2:
                if label != 'healthy':
                    label = 'high'
            sub_labels[row] = self.diagnosis_dict[label]
        self.subtissue_labels = sub_labels
        
        tissue_labels_numbers = np.zeros((tissue_label.shape[0]))
        for row in range(0, tissue_label.shape[0]):
            label = tissue_label[row].decode('UTF-8')
            if num_classes == 2:
                if label != 'healthy':
                    label = 'high'
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

            
        
        

        