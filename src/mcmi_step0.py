#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:45:55 2021

@author: stephenschmidt
"""

from load_train_data import MSITrainData

label_data_filename = "final_annotation_complete_cores_2.tabular"
msi_data_filename = "msi.h5"

h5_data = MSITrainData(total_pixel_name=label_data_filename, data_name=msi_data_filename)