#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:10:36 2021

@author: stephenschmidt
"""

from load_data import H5MSI, SmallROI
import os
import collections
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt


def model_one_feature(feature_num = 9):
    smallROI = SmallROI()
    
    healthy_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['healthy'])
    high_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['high'])
    low_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['low'])
    ca_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['CA']) 
    
    healthy_tissue_values = smallROI.spec[healthy_locations]
    high_tissue_values = smallROI.spec[high_locations]
    low_tissue_values = smallROI.spec[low_locations]
    ca_tissue_values = smallROI.spec[ca_locations]
    
    ca_cores = np.unique(smallROI.core[ca_locations])
    core_chosen = ca_cores[7]
    
    # 1, 7,5, 9, 10 is good ish for CA
    
    single_ca_tissue_locations = np.where(smallROI.core==core_chosen)# choose first tissue available
    single_ca_tissue_count = single_ca_tissue_locations[0].shape[0]
    
    print('Total Values Count: ', single_ca_tissue_count)
    print('Core Number: ', core_chosen)
    
    single_ca_tissue_healthy_locations = np.where(smallROI.subtissue_labels[single_ca_tissue_locations] == smallROI.diagnosis_dict['healthy'])  
    single_ca_tissue_healthy_locations = single_ca_tissue_locations[0][single_ca_tissue_healthy_locations]
    single_ca_tissue_healthy_values = smallROI.spec[single_ca_tissue_healthy_locations]
    single_ca_tissue_healthy_values = single_ca_tissue_healthy_values[:, feature_num]
    
    single_ca_tissue_ca_locations = np.where(smallROI.subtissue_labels[single_ca_tissue_locations] == smallROI.diagnosis_dict['CA'])  
    single_ca_tissue_ca_locations = single_ca_tissue_locations[0][single_ca_tissue_ca_locations]
    single_ca_tissue_ca_values = smallROI.spec[single_ca_tissue_ca_locations]
    single_ca_tissue_ca_values = single_ca_tissue_ca_values[:, feature_num]
    
    single_ca_tissue_healthy_count = single_ca_tissue_healthy_values.shape[0]    
    single_ca_tissue_ca_count = single_ca_tissue_ca_values.shape[0]
    
    
    
    