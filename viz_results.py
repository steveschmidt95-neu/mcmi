#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:08:00 2021

@author: stephenschmidt
"""

import h5py
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import tensorflow.compat.v1 as v1
v1.disable_eager_execution()
import time
import os
from net1_MIL import MSInet1
from mil import MIL

class SmallROIViz():
    
    def __init__(self, h5_name = 'smallROI.h5'):
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
    


class MIL_Data_Only():
   def  __init__(self, num_classes=4):
       
        self.smallROI = SimData()
        #self.smallROI = SmallROIViz()      # Change here to SmallROIViz()
        self.smallROI.split_cores()
        self.smallROI.cores_list = [1,2,3,9]    # for sim data
        self.num_classes = num_classes
        #smallROI.cores_list = [7, 10,6, 4,]
        self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 0}
        self.diagnosis_dict_reverse = {1: 'high', 2: 'CA', 3: 'low', 0:'healthy'}
        
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
    
    
   def viz_single_core_true(self, core):
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
        
        cmap = matplotlib.colors.ListedColormap(['black', 'red', 'blue', 'yellow', 'white'])
        color_bar_labels = ['healthy', 'CA', 'low', 'high']
        bounds = [0, 1, 2, 3, 4,5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        for location in range(0, self.core_true_sub_labels[core].shape[0]):
            label = self.core_true_sub_labels[core][location]
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
        cbar = plt.colorbar(cmap=cmap,norm=norm,boundaries=bounds,ticks=[0,1,2,3])
        
        for j, clabel in enumerate(color_bar_labels):
            #cbar.ax.text(.5, (2 * j + 1)/2, clabel, ha='center', va='center')
            cbar.ax.text(7,  (2 * j + 1)/2, clabel, va='center')
        filename = 'Images/TrueSim'+ str(int(core)) + '.png'
        print(filename)
        
        plt.savefig(filename, pad_inches=0)
        plt.clf()
  
   def save_im_all_cores(self):
        for core in self.smallROI.cores_list:
            print(core)
            self.viz_single_core_true(core)


    
    
#num_classes = 4

mil_data = MIL_Data_Only(num_classes=4)
mil_data.save_im_all_cores()