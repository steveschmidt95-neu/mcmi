#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 20:20:18 2021

@author: stephenschmidt
"""

import os
import sys
import h5py
import numpy as np
import collections
import matplotlib.pyplot as plt

class ROI_Total_Data():
    
    def __init__(self, msi_data, sliced_roi_positions, tuple_positions):
        
        self.roi_num = str(int(sliced_roi_positions[0, 3]))
        self.tma_num = str(int(sliced_roi_positions[0, 2]))
        self.data = None
        self.data_folder = os.path.join(os.path.dirname(os.getcwd()), 'Data')
        
        self.data = np.zeros((sliced_roi_positions.shape[0], msi_data.shape[1]))
        self.labels = np.zeros((sliced_roi_positions.shape[0])) # first spot is core label, second is subtissue
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
                    self.labels[line] = sliced_roi_positions[line, 4]
                        
            if not found_msi_data:
                #assert False # no msi data found for this
                miss_count +=1
                miss_lines.append(line)
                
        print('Missing Values: ', miss_count)
        self.labels = np.delete(self.labels, miss_lines, 0)
        self.data = np.delete(self.data, miss_lines, 0)
        assert self.labels.shape[0] == self.data.shape[0]
        
        
    def save_h5(self):
        
        data_filename = os.path.join(self.data_folder, 'ROI' + self.roi_num + "TMA" + self.tma_num + 'train.hdf5')
        assert (len(self.data.shape)==2)
        with h5py.File(data_filename, "w") as f:
            dset = f.create_dataset("roi" + self.roi_num + "train_data", data=self.data, dtype='f')
            
        labels_filename = os.path.join(self.data_folder, 'ROI'+ self.roi_num +  "TMA" + self.tma_num + 'train_Labels' + '.hdf5')
        with h5py.File(labels_filename, "w") as f:
            dset = f.create_dataset("roi" + self.roi_num + "train_labels", data=self.labels, dtype='f')


class MSITrainData():
    
    def __init__(self, data_name = 'msi.h5', sub_tab_name = 'subtissue_labels.tabular',
                 total_pixel_name = 'final_annotation_complete_cores_2.tabular'):
        
        self.data_folder = os.path.join(os.path.dirname(os.getcwd()), 'OriginalData')
        self.data_path = os.path.join(self.data_folder, data_name)
        #self.sub_tab_path = os.path.join(self.data_folder, sub_tab_name)
        self.total_pixel_name = os.path.join(self.data_folder, total_pixel_name)
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
        for line in open(self.total_pixel_name):
            line_data = line.split()
            line_count += 1
            
        position_data = np.zeros((line_count-1, 5))
        line_count = 0
        for line in open(self.total_pixel_name):
            if line_count == 0:
                line_count+=1
                continue
            line_data = line.split()   
            
            position_data[line_count-1, 0] = int(line_data[0]) # X
            position_data[line_count-1, 1] = int(line_data[1]) # Y
            position_data[line_count-1, 2] = int(line_data[2][-1]) # TMANum
            position_data[line_count-1, 3] = int(line_data[3][3:]) # SlideNum
            position_data[line_count-1, 4] = self.diagnosis_dict[line_data[4]] # Label

            line_count+=1
        self.total_subtissue_position_data = position_data
        
        # put subtissue entries into a matrix
        line_count = 0
        for line in open(self.total_pixel_name):
            line_data = line.split()
            line_count += 1
        
        #build each ROI one at a time
        print('Number of ROIs', np.max(self.total_subtissue_position_data[:, 3]))
        num_rois = int(np.max(position_data[:, 3]))
        
        
        for roi in range(1, num_rois):
            for tma in range(1,3):
                print('-'*20)
                print('ROI NUM: ', roi)
                print("TMA: ", tma)
                
                sliced_positions = self.total_subtissue_position_data[np.where(self.total_subtissue_position_data[:,3] == roi)]
                sliced_positions = sliced_positions[np.where(sliced_positions[:,2] == tma)]
                
                
                if sliced_positions.shape[0] == 0: # some ROI's dont exist
                    print("No ROI values")
                    continue
                
                roi_data = ROI_Total_Data(self.msispec, sliced_positions, self.position)
                roi_data.save_h5()    

class H5MSI_Train():
    
    def __init__(self):
        self.data_folder = os.path.join(os.path.dirname(os.getcwd()), 'Data')
        self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        
        h5_files = os.listdir(self.data_folder)
        h5_files = [elem for elem in h5_files if elem.endswith('.hdf5')]
        h5_files = [elem for elem in h5_files if 'train' in elem]
        self.num_rois = len(h5_files)/2
        self.data_files = {}
        self.train_data = {}
        self.num_classes = 4
        self.data_is_flat = False
        self.flat_labels = {}
        self.cores_list = []
        
        # Read in all the h5 files
        for h5_file in h5_files:
            roi_path = os.path.join(self.data_folder, h5_file)
            
            idx = h5_file.find('.')
            dict_key = h5_file[0:idx]

            with h5py.File(roi_path, "r") as hf:
                dname = list(hf.keys())[0]
                n1 = hf.get(dname)    
                n1_array = np.copy(n1)
                
                self.data_files[dict_key] = n1_array
        
        # seperate them into matching data and label files
        for key in self.data_files.keys():
            if not ('Labels' in key):
                self.cores_list.append(key)
                self.train_data[key] = self.data_files[key]
                self.train_data[key + '_Labels'] = self.data_files[key + '_Labels']
 
        
    # Turn the labels from having multiple class labels into one hot    
    def one_hot_labels(self):
        
        for key in self.train_data.keys():
            if 'Labels' in key:
                prev_labels = self.train_data[key] = self.train_data[key]
                flat_labels = np.zeros((prev_labels.shape[0], 2))
                
                if prev_labels[0] == 4:
                    self.train_data[key][:] = 0
                
                if (prev_labels[0] == 0): # Healthy
                    flat_labels[:, 0] = 1
                else:
                    flat_labels[:, 1] = 1 # All other 3 labels
                
                self.flat_labels[key] = flat_labels
        
        count_pos = 0
        count_neg = 0
        for key in self.train_data.keys():
            if 'Labels' in key:
                count_pos +=np.sum(self.flat_labels[key] [:,1])
                count_neg +=np.sum(self.flat_labels[key] [:,0])
                
            
        print("Initial Pos: ", count_pos)
        print("Initial Neg: ", count_neg)



#orig = MSITrainData()

#h5 = H5MSI_Train()            
#h5.one_hot_labels()
