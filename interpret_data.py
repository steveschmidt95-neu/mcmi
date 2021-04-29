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
import matplotlib.patches as mpatches
from scipy.stats import norm, truncnorm
from viz_results import MIL_Data_Only


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
    
    
    ax1.hist(single_ca_tissue_values, bins=bins, range = (0,10))
    ax1.text(2, 10, 'Total Count: ' + str(single_ca_tissue_count), fontsize=12)
    ax1.set_title('Single Tissue Values')
    
    ax2.hist(single_ca_tissue_ca_values, bins=bins, range = (0,10))
    ax2.text(3, 10, 'CA Count: ' + str(single_ca_tissue_ca_count), fontsize=12)
    ax2.set_title('CA')
    
    plt.show()
    
    # Plot Total distribution for single tissue against Healthy
    # ----------------------------------------------------------
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True)  
    fig.suptitle('Single Tissue Distributions')
    
    ax1.hist(single_ca_tissue_values, bins=bins, range = (0,10))
    ax1.text(2, 10, 'Total Count: ' + str(single_ca_tissue_count), fontsize=12)
    ax1.set_title('Single Tissue Values')
    
    ax2.hist(single_ca_tissue_healthy_values, bins=bins, range = (0,10))
    ax2.text(2, 10, 'Healthy Count: ' + str(single_ca_tissue_healthy_count), fontsize=12)
    ax2.set_title('Healthy')
    
    
    healthy_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['healthy'])    
    healthy_tissue_values = smallROI.spec[healthy_locations]
    healthy_tissue_feature_valeus = healthy_tissue_values[:, feature_num]
    
    print('Mean Healthy: ', np.mean(healthy_tissue_feature_valeus))
    print('Mean CA: ', np.mean(single_ca_tissue_ca_values))
    print(single_ca_tissue_ca_values)
    plt.show()
    
    
def pdf_single_tissue(feature_num=9):
    smallROI = SmallROI()
    
    healthy_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['healthy'])
    ca_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['CA']) 
    
    healthy_tissue_values = smallROI.spec[healthy_locations]
    ca_tissue_values = smallROI.spec[ca_locations]
    healthy_tissue_values = healthy_tissue_values[:, feature_num]
    
    ca_cores = np.unique(smallROI.core[ca_locations])
    core_chosen = ca_cores[7]
    
    # 1, 7,5, 9, 10 is good ish for CA
    
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
    
    
    #fig, ax = plt.subplots(1, 1)
    
    
    low = 0
    upp = np.inf
    
    mean, sd = norm.fit(healthy_tissue_values)
    
    #trunc_norm=truncnorm(a=low, b=upp, loc=mean, scale=sd)
    
    trunc_norm = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    
    #x = np.linspace(truncnorm.ppf(0.01, low, upp),truncnorm.ppf(0.99, low, upp), 100)
    x = np.linspace(0, 8, 100)
    plt.plot(x, trunc_norm.pdf(x), 'k-', lw=2, label='frozen pdf')
    
    plt.hist(single_ca_tissue_ca_values, bins=11, range = (0,7), density=True)
    plt.title('Ca Values Against Truncated Normal')
    
    
    plt.plot(x, trunc_norm.pdf(x), 'k-', lw=2, label='frozen pdf')
    plt.hist(single_ca_tissue_ca_values, bins=11, range = (0,7), density=True)
    plt.title('Ca Values Against Truncated Normal')
    
    
    
    
    
def single_tissue_select_data_one_feature(feature_num=9, k=10):
    smallROI = SmallROI()
    
    healthy_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['healthy'])
    ca_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['CA']) 
    
    healthy_tissue_values = smallROI.spec[healthy_locations]
    ca_tissue_values = smallROI.spec[ca_locations]
    healthy_tissue_values = healthy_tissue_values[:, feature_num]
    
    ca_cores = np.unique(smallROI.core[ca_locations])
    core_chosen = ca_cores[7]
    
    # 1, 7,5, 9, 10 is good ish for CA
    
    single_ca_tissue_locations = np.where(smallROI.core==core_chosen)# choose first tissue available
    single_ca_tissue_count = single_ca_tissue_locations[0].shape[0]
    
    print('Total Values Count: ', single_ca_tissue_count)
    print('Core Number: ', core_chosen)
    
    single_ca_tissue_values = smallROI.spec[single_ca_tissue_locations]
    single_ca_tissue_values = single_ca_tissue_values[:, feature_num]
    single_ca_tissue_labels = smallROI.subtissue_labels[single_ca_tissue_locations]
    
    low = 0
    upp = np.inf
    
    mean, sd = norm.fit(healthy_tissue_values)
    trunc_norm = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    
    pdf_values = trunc_norm.pdf(single_ca_tissue_values)
    idx_sorted_pdf = np.argsort(pdf_values)
    pdf_sorted = pdf_values[idx_sorted_pdf]
    tissue_pdf_sorted_locs= single_ca_tissue_locations[0][idx_sorted_pdf]
    k_labels = smallROI.subtissue_labels[tissue_pdf_sorted_locs]
    
    
    print(k_labels)
    assert False
    
    ca_labels_locs = np.where(k_labels==2)
    healthy_labels = np.where(k_labels ==0)
    
    print()
    
    ca_labels_pdf_values = pdf_sorted[ca_labels_locs]
    healthy_labels_pdf_values = pdf_sorted[healthy_labels]
    
    print(pdf_sorted)
    print(k_labels)
    
    fig, ax = plt.subplots()
    
    
    scatter = ax.scatter(healthy_labels_pdf_values, np.ones((healthy_labels_pdf_values.shape[0])), c='blue', marker='|')
    scatter = ax.scatter(ca_labels_pdf_values, np.zeros((ca_labels_pdf_values.shape[0])), c='red', marker='|')
    
    red_patch = mpatches.Patch(color='red', label='CA')
    blue_patch = mpatches.Patch(color='blue', label='THealthy')
    
    ax.legend(handles=[red_patch, blue_patch])
    

def plot_class_imbalance_subtissue():
    milData = MIL_Data_Only()
    
    data_count = [0, 0, 0, 0]
    
    for core in milData.smallROI.cores_list:
        print('Core: ', core)
        core_sub = milData.core_true_sub_labels[core]
        for loc in range(0, core_sub.shape[0]):
            sub_val = int(core_sub[loc])
            data_count[sub_val] +=1
    
    print("Core Count: ", len(milData.smallROI.cores_list))
    x_axis = [0, 1, 2, 3]
    labels = ['healthy', 'high', 'CA', 'low']
    print(data_count)
    
    fig, ax = plt.subplots()
    
    ax.bar(x_axis, data_count)
    ax.set_ylabel('Subtissue Locations Count')
    ax.set_title('Dataset Subtissue Distribution')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(labels)
    
    plt.show()
    
def plot_class_imbalance_tissue():
    milData = MIL_Data_Only()
    data_count = [0, 0, 0, 0]
    
    for core in milData.smallROI.cores_list:
        print('Core: ', core)
        core_sub = milData.core_pred_sub_labels[core]
        for loc in range(0, core_sub.shape[0]):
            sub_val = int(core_sub[loc])
            data_count[sub_val] +=1
    
    print("Core Count: ", len(milData.smallROI.cores_list))
    x_axis = [0, 1, 2, 3]
    labels = ['healthy', 'high', 'CA', 'low']
    print(data_count)
    
    fig, ax = plt.subplots()
    
    ax.bar(x_axis, data_count)
    ax.set_ylabel('Tissue Locations Count')
    ax.set_title('Dataset Tissue Distribution')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(labels)
    
    plt.show()
    
    
        
    
    
    
    
    
#single_tissue_select_data_one_feature(feature_num=9)    
plot_class_imbalance_tissue()

#plot_single_distribution()
#histo_single_feature(feature_num=9)
#histo_single_tissue(feature_num=208)
#pdf_single_tissue(feature_num=208)

    
    
    
    