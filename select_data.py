#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:02:25 2021

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

def kl(m0, s0, m1, s1):
    term1 = np.log(s1/s0)
    term2 =((s0**2)+((s0-s1)**2)) / (2*(s1)**2)
    
    return(term1 + term2 - .5)
    

def smallestN_indices_argparitition(a, N, maintain_order=True):
    idx = np.argpartition(a.ravel(),N)[:N]
    if maintain_order:
        idx = idx[a.ravel()[idx].argsort()]
    return np.stack(np.unravel_index(idx, a.shape)).T


def choose_x_features(num_features = 3):
    smallROI = SmallROI()
    
    healthy_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['healthy'])
    ca_locations = np.where(smallROI.tissue_labels==smallROI.diagnosis_dict['CA']) 
    
    healthy_tissue_values = smallROI.spec[healthy_locations]
    ca_tissue_values = smallROI.spec[ca_locations]
    
    upp = np.inf
    low = 0
    
    ca_cores = np.unique(smallROI.core[ca_locations])
    core_chosen = ca_cores[7]
    print('Core Number: ', core_chosen)
    # 1, 7,5, 9, 10 is good ish for CA
        
    single_ca_tissue_locations = np.where(smallROI.core==core_chosen)# choose first tissue available
    single_ca_tissue_count = single_ca_tissue_locations[0].shape[0]
    print('Total Values Count: ', single_ca_tissue_count)
    single_ca_tissue_values = smallROI.spec[single_ca_tissue_locations]
    
    # Select The features to use
    feature_differences = np.zeros((healthy_tissue_values.shape[1]))
    for feature_num in range(0, healthy_tissue_values.shape[1]):
        healthy_tissue_feature = healthy_tissue_values[:, feature_num]
        mean, sd = norm.fit(healthy_tissue_feature)
        
        single_ca_tissue_feature = single_ca_tissue_values[:, feature_num]
        mean_ca, sd_ca = norm.fit(single_ca_tissue_feature)
        
        diff = mean_ca - mean
        #diff =  kl(mean, sd, mean_ca, sd_ca)
        #print('Feature: ', feature_num, ' Diff: ', diff)
        
        if diff < 0:
            feature_differences[feature_num] = 0
        else:
            feature_differences[feature_num] = diff

    # Choose K Best Features to use
    # Choose k largest differences
    print("Choosing " + str(num_features) +' Features')
    locs = np.argpartition(feature_differences, -num_features)
    chosen_features = locs[-num_features:]
    chosen_features = [9,10]
    print("Chosen Features: ", chosen_features)    
    
    # Evauluate the TRuncated norm PDF for these values
    tissue_pdf = np.ones((single_ca_tissue_values.shape[0], len(chosen_features)))
    num_higher = 0
    num_lower = 0
    for feature_idx, feature_num in enumerate(chosen_features):
        healthy_tissue_feature = healthy_tissue_values[:, feature_num]
        mean, sd = norm.fit(healthy_tissue_feature)
        trunc_norm = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
        
        single_ca_tissue_feature = single_ca_tissue_values[:, feature_num]
        mean_ca, sd_ca = norm.fit(single_ca_tissue_feature)
        pdf_values = trunc_norm.pdf(single_ca_tissue_feature)
        
        tissue_pdf[:,feature_idx] = pdf_values
    
    '''
    # Use this for summing All features
    tissue_pdf = tissue_pdf.sum(axis = 1)
    idx_sorted_pdf = np.argsort(pdf_values)
    pdf_sorted = pdf_values[idx_sorted_pdf]
    tissue_pdf_sorted_locs= single_ca_tissue_locations[0][idx_sorted_pdf]
    k_labels = smallROI.subtissue_labels[tissue_pdf_sorted_locs]
    print(k_labels)
    
    '''
    # Use this one for selecting them indepently without summing
    smallest_indeces = smallestN_indices_argparitition(tissue_pdf, 100)
    for idx in range(smallest_indeces.shape[0]):
        loc = smallest_indeces[idx]
    
    selected_examples = smallest_indeces[:, 0]
    tissue_pdf_sorted_locs= single_ca_tissue_locations[0][selected_examples]
    k_labels = smallROI.subtissue_labels[tissue_pdf_sorted_locs]
    print(k_labels)

    
    
    
choose_x_features(num_features = 593)