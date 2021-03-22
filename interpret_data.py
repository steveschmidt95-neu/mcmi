#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:25:06 2021

@author: stephenschmidt
"""

from load_data import H5MSI, SmallROI
import os
import collections
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt


def average_values_plot():
    msi = H5MSI()
    msi.flatten_data()
    histo_count = np.zeros((5))
    x_axis = [1, 2, 3, 4]
    
    print(msi.diagnosis_dict)
    
    high = msi.flat_train[np.where(msi.flat_train_labels==msi.diagnosis_dict['high'])]
    CA = msi.flat_train[np.where(msi.flat_train_labels==msi.diagnosis_dict['CA'])]
    low = msi.flat_train[np.where(msi.flat_train_labels==msi.diagnosis_dict['low'])]
    healthy = msi.flat_train[np.where(msi.flat_train_labels==msi.diagnosis_dict['healthy'])]
    
    
    total = high.shape[0] + CA.shape[0] + low.shape[0] + healthy.shape[0]
    y_count = [high.shape[0] ,CA.shape[0] , low.shape[0] , healthy.shape[0]]
    
    print(y_count)
    
    high_mean = np.mean(high, axis=0)
    CA_mean = np.mean(CA, axis=0)
    low_mean = np.mean(low, axis=0)
    healthy_mean = np.mean(healthy, axis=0)
    
    x_axis = np.arange(0, high.shape[1])
    
    fig = plt.figure()    
    
    plt.subplot(2, 2, 1)
    plt.plot(x_axis, high_mean)
    plt.title('High')
    plt.ylim(0,25)
    
    plt.subplot(2, 2, 2)
    plt.plot(x_axis, CA_mean)
    plt.title('CA')
    plt.ylim(0,25)
    
    plt.subplot(2, 2, 3)
    plt.plot(x_axis, low_mean)
    plt.title('Low')
    plt.ylim(0,25)
    
    plt.subplot(2, 2, 4)
    plt.plot(x_axis, healthy_mean)
    plt.title('Healthy')
    plt.ylim(0,25)

    plt.show()
   
def plot_single_distribution():
    smallROI = SmallROI()
    
    healthy_sub_locations = np.where(smallROI.subtissue_labels==smallROI.diagnosis_dict['healthy'])
    high_sub_locations = np.where(smallROI.subtissue_labels==smallROI.diagnosis_dict['high'])
    low_sub_locations = np.where(smallROI.subtissue_labels==smallROI.diagnosis_dict['low'])
    ca_sub_locations = np.where(smallROI.subtissue_labels==smallROI.diagnosis_dict['CA']) 
    
    print('Healthy Subtissue Labels Count: ', len(healthy_sub_locations[0]))
    print('High Subtissue Labels Count: ', len(high_sub_locations[0]))
    print('Low Subtissue Labels Count: ', len(low_sub_locations[0]))
    print('Ca Subtissue Labels Count: ', len(ca_sub_locations[0]))
    
    healthy_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['healthy'])
    high_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['high'])
    low_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['low'])
    ca_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['CA']) 
    
    print()
    print('Healthy Tissue Labels Count: ', len(healthy_locations[0]))
    print('High Tissue Labels Count: ', len(high_locations[0]))
    print('Low Tissue Labels Count: ', len(low_locations[0]))
    print('Ca Tissue Labels Count: ', len(ca_locations[0]))
    
    healthy_sub_values = smallROI.spec[healthy_sub_locations]
    high_sub_values = smallROI.spec[high_sub_locations]
    low_sub_values = smallROI.spec[low_sub_locations]
    ca_sub_values = smallROI.spec[ca_sub_locations]
    
    healthy_tissue_values = smallROI.spec[healthy_locations]
    high_tissue_values = smallROI.spec[high_locations]
    low_tissue_values = smallROI.spec[low_locations]
    ca_tissue_values = smallROI.spec[ca_locations]
    
    healthy_sub_mean = np.mean(healthy_sub_values,axis=0)
    high_sub_mean = np.mean(high_sub_values,axis=0)
    low_sub_mean = np.mean(low_sub_values,axis=0)
    ca_sub_mean = np.mean(ca_sub_values,axis=0)
    
    healthy_tissue_mean = np.mean(healthy_tissue_values,axis=0)
    high_tissue_mean = np.mean(high_tissue_values,axis=0)
    low_tissue_mean = np.mean(low_tissue_values,axis=0)
    ca_tissue_mean = np.mean(ca_tissue_values,axis=0)
    
    
    x_axis = np.arange(0, ca_tissue_values.shape[1])
    fig = plt.figure()    
    
    plt.subplot(2, 2, 1)
    plt.plot(x_axis, high_tissue_mean)
    plt.title('High')
    plt.ylim(0,25)
    
    plt.subplot(2, 2, 2)
    plt.plot(x_axis, ca_tissue_mean)
    plt.title('CA')
    plt.ylim(0,25)
    
    plt.subplot(2, 2, 3)
    plt.plot(x_axis, low_tissue_mean)
    plt.title('Low')
    plt.ylim(0,25)
    
    plt.subplot(2, 2, 4)
    plt.plot(x_axis, healthy_tissue_mean)
    plt.title('Healthy')
    plt.ylim(0,25)
    
    fig.suptitle("Tissue Distributions", fontsize=14)    
    
    

def histo_single_feature(feature_num=9):
    smallROI = SmallROI()
    
    healthy_sub_locations = np.where(smallROI.subtissue_labels==smallROI.diagnosis_dict['healthy'])
    high_sub_locations = np.where(smallROI.subtissue_labels==smallROI.diagnosis_dict['high'])
    low_sub_locations = np.where(smallROI.subtissue_labels==smallROI.diagnosis_dict['low'])
    ca_sub_locations = np.where(smallROI.subtissue_labels==smallROI.diagnosis_dict['CA']) 
    
    healthy_sub_count = len(healthy_sub_locations[0])
    high_sub_count = len(high_sub_locations[0])
    low_sub_count = len(low_sub_locations[0])
    ca_sub_count = len(ca_sub_locations[0]) 
    
    print('Healthy Subtissue Labels Count: ', healthy_sub_count)
    print('High Subtissue Labels Count: ', high_sub_count)
    print('Low Subtissue Labels Count: ', low_sub_count)
    print('Ca Subtissue Labels Count: ', ca_sub_count)
    
    healthy_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['healthy'])
    high_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['high'])
    low_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['low'])
    ca_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['CA']) 
    
    healthy_tissue_count = len(healthy_locations[0])
    high_tissue_count = len(high_locations[0])
    low_tissue_count = len(low_locations[0])
    ca_tissue_count = len(ca_locations[0])
    
    print()
    print('Healthy Tissue Labels Count: ', healthy_tissue_count)
    print('High Tissue Labels Count: ', high_tissue_count)
    print('Low Tissue Labels Count: ', low_tissue_count)
    print('Ca Tissue Labels Count: ', ca_tissue_count)
    
    
    healthy_sub_values = smallROI.spec[healthy_sub_locations]
    high_sub_values = smallROI.spec[high_sub_locations]
    low_sub_values = smallROI.spec[low_sub_locations]
    ca_sub_values = smallROI.spec[ca_sub_locations]
    
    healthy_tissue_values = smallROI.spec[healthy_locations]
    high_tissue_values = smallROI.spec[high_locations]
    low_tissue_values = smallROI.spec[low_locations]
    ca_tissue_values = smallROI.spec[ca_locations]
    
    healthy_sub_mean = np.mean(healthy_sub_values,axis=0)
    high_sub_mean = np.mean(high_sub_values,axis=0)
    low_sub_mean = np.mean(low_sub_values,axis=0)
    ca_sub_mean = np.mean(ca_sub_values,axis=0)
    
    healthy_tissue_mean = np.mean(healthy_tissue_values,axis=0)
    high_tissue_mean = np.mean(high_tissue_values,axis=0)
    low_tissue_mean = np.mean(low_tissue_values,axis=0)
    ca_tissue_mean = np.mean(ca_tissue_values,axis=0)
    
    healthy_sub_features = healthy_sub_values[:, feature_num]
    ca_sub_features = ca_sub_values[:, feature_num]
    
    healthy_tissue_features = healthy_tissue_values[:, feature_num]
    ca_tissue_features = ca_tissue_values[:, feature_num]
    
        
    # Plot subtissue distributions
    # ----------------------------------------------------------
    
    print('Healthy Sub Mean Value: ', healthy_sub_mean[feature_num])
    print('CA Sub Mean Value: ', ca_sub_mean[feature_num])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True)  
    fig.suptitle('Subtissue Distributions')
    bins = 30
    
    ax1.hist(healthy_sub_features, bins=bins, range = (0,7))
    ax1.text(2, 200, 'Healthy Count: ' + str(healthy_sub_count), fontsize=12)
    ax1.set_title('Healthy')
    
    ax2.hist(ca_sub_features, bins=bins, range = (0,7))
    ax2.text(3, 200, 'CA Count: ' + str(ca_sub_count), fontsize=12)
    ax2.set_title('CA')
    
    # -----------------------------------------------------------
    # Now plot total features distributions
    
    print('Healthy Total Mean Value: ', healthy_tissue_mean[feature_num])
    print('CA Total Mean Value: ', ca_tissue_mean[feature_num])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True) 
    fig.suptitle('Core Tissue Distributions')
    bins = 30
    
    ax1.hist(healthy_tissue_features, bins=bins, range = (0,7))
    ax1.text(2, 200, 'Healthy Count: ' + str(healthy_tissue_count), fontsize=12)
    ax1.set_title('Healthy')

    
    ax2.hist(ca_tissue_features, bins=bins, range = (0,7))
    ax2.text(3, 200, 'CA Count: ' + str(ca_tissue_count), fontsize=12)
    ax2.set_title('CA')
    

def histo_single_tissue(feature_num=9):
    smallROI = SmallROI()
    
    healthy_sub_locations = np.where(smallROI.subtissue_labels==smallROI.diagnosis_dict['healthy'])
    high_sub_locations = np.where(smallROI.subtissue_labels==smallROI.diagnosis_dict['high'])
    low_sub_locations = np.where(smallROI.subtissue_labels==smallROI.diagnosis_dict['low'])
    ca_sub_locations = np.where(smallROI.subtissue_labels==smallROI.diagnosis_dict['CA']) 
    
    
    healthy_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['healthy'])
    high_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['high'])
    low_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['low'])
    ca_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['CA']) 
    
    ca_cores = np.unique(smallROI.core[ca_locations])
    core_chosen = ca_cores[7]
    
    # 1, 7,5, 9, 10 is good ish fro CA
    
    single_ca_tissue_locations = np.where(smallROI.core==core_chosen)# choose first tissue available
    single_ca_tissue_count = single_ca_tissue_locations[0].shape[0]
    
    print('Total Values Count: ', single_ca_tissue_count)
    print('Core Number: ', core_chosen)
    
    single_ca_tissue_values = smallROI.spec[single_ca_tissue_locations]
    single_ca_tissue_values = single_ca_tissue_values[:, feature_num]
    
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
    
    
    bins = 10
    
    
    # Plot Total distribution for single tissue against CA
    # ----------------------------------------------------------
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True)  
    fig.suptitle('Single Tissue Distributions')
    
    
    ax1.hist(single_ca_tissue_values, bins=bins, range = (0,7))
    ax1.text(2, 10, 'Total Count: ' + str(single_ca_tissue_count), fontsize=12)
    ax1.set_title('Single Tissue Values')
    
    ax2.hist(single_ca_tissue_ca_values, bins=bins, range = (0,7))
    ax2.text(3, 10, 'CA Count: ' + str(single_ca_tissue_ca_count), fontsize=12)
    ax2.set_title('CA')
    
    plt.show()
    
    # Plot Total distribution for single tissue against Healthy
    # ----------------------------------------------------------
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True)  
    fig.suptitle('Single Tissue Distributions')
    
    ax1.hist(single_ca_tissue_values, bins=bins, range = (0,7))
    ax1.text(2, 10, 'Total Count: ' + str(single_ca_tissue_count), fontsize=12)
    ax1.set_title('Single Tissue Values')
    
    ax2.hist(single_ca_tissue_healthy_values, bins=bins, range = (0,7))
    ax2.text(2, 10, 'Healthy Count: ' + str(single_ca_tissue_healthy_count), fontsize=12)
    ax2.set_title('Healthy')
    
    plt.show()
    
    
    
    
    
    
    


#plot_single_distribution()
#histo_single_feature(feature_num=9)
histo_single_tissue(feature_num=9)


    
    
    
    