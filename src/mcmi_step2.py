#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:37:21 2021

@author: stephenschmidt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:12:13 2021

@author: stephenschmidt


uses the undertraining on the total dataset and then applies it to

"""



import numpy as np
import tensorflow.compat.v1 as v1
v1.disable_eager_execution()
import os
from direct_cnn_dataset import DirectCNNDataset
from load_train_data import H5MSI_Train
from net1_adam import MSInet1
import matplotlib.pyplot as plt
import matplotlib
import h5py


def single_to_one_hot(labels, num_classes):
        #diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        # shifted to {'high': 0, 'CA': 1, 'low': 2, 'healthy': 3}
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for hot_class in range(0, num_classes):
        class_locations = np.where(labels == hot_class)
        one_hot_labels[class_locations, hot_class] = 1
    return(one_hot_labels)

def one_hot_probability_to_single_label(pred_labels, num_classes):    
    argmax_labels = np.argmax(pred_labels, axis=1)
    return(argmax_labels)


class MIL():
    
    def __init__(self,  input_file = 'MICNN_Out', fc_units = 50, num_classes=4, width1=38, width2=18, width3=16, filters_layer1=12, 
                 filters_layer2=24, filters_layer3=48, batch_size=4, lr=.001, keep_prob=.8):
        
        self.train_dataset = DirectCNNDataset(num_classes=4, train_data_location=input_file)
        self.num_classes = num_classes
                    
        self.h5_out_path = os.path.join(os.path.dirname(os.getcwd()), 'FinalOutputPredictions')
        self.image_out_path = os.path.join(os.path.dirname(os.getcwd()), 'FinalOutputImages')
        self.data_folder = os.path.join(os.path.dirname(os.getcwd()), 'Data')

        self.train_data_total = H5MSI_Train()
        self.train_data_total.one_hot_labels()
        
        self.train_core_spec = {}
        self.train_core_pred_sub_labels = {}
        self.train_positions = {}
        self.test_core_true_label = {}
                        
        for core in self.train_data_total.cores_list:
            if 'Label' not in core and 'position' not in core:
                self.train_core_spec[core] = self.train_data_total.train_data[core]
                self.train_core_pred_sub_labels[core] = np.zeros((self.train_core_spec[core].shape[0]))
                self.train_positions[core] = self.get_positions_for_core(core)
                self.test_core_true_label[core] = int(self.train_data_total.train_data[core +'_Labels'][0])
        
        self.diagnosis_dict =         {'high': 1, 'CA': 2, 'low': 3, 'healthy': 0}
        self.diagnosis_dict_reverse = {1: 'high', 2: 'CA', 3: 'low', 0:'healthy'}
        
        if num_classes == 2:
            self.diagnosis_dict =         {'high': 1, 'healthy': 0}
            self.diagnosis_dict_reverse = {1: 'high',  0:'healthy'}
            
        self.batch_size = batch_size
        self.lr = lr
        self.keep_prob = keep_prob
        
    
        self.sample_shape = int(self.train_dataset.spec_dict[0].shape[1])

        self.net1 = MSInet1(data_shape = self.sample_shape, fc_units=fc_units, num_classes=num_classes, width1=width1, width2=width2, width3=width3, filters_layer1=filters_layer1, filters_layer2=filters_layer2, filters_layer3=filters_layer3, batch_size=batch_size*self.num_classes,lr=lr)
        self.net1.build_graph()
    
    
    def check_batch_size(self):
        for class_label in range(0, self.num_classes):
            for class_label in range(0, self.num_classes):
                label_spec_size = self.train_dataset.spec_dict[class_label].shape[0]
                if label_spec_size < self.batch_size*self.num_classes:
                    print("Class: ", class_label, " Has ", label_spec_size, " Inputs")
                    assert False # Reduce batch Size
                    
        for core in self.train_data_total.cores_list:
            if ((self.num_classes * self.batch_size) > self.train_core_spec[core].shape[0]):
                print("Core: ", core, " Has ", self.train_core_spec[core].shape[0], " Inputs")
                assert False # Reduce batch Size
                
            
                
    
    def get_next_batch(self):
        
        random_batch = np.zeros((self.batch_size*self.num_classes,self.train_dataset.spec_dict[0].shape[1]))
        labels = np.zeros((self.batch_size*self.num_classes, self.num_classes))
        
        for class_label in range(0, self.num_classes):
            class_spec = self.train_dataset.spec_dict[class_label]
            random_batch_indeces = np.random.choice(class_spec.shape[0], self.batch_size)
            
            random_class_batch = class_spec[random_batch_indeces, :]
            random_batch[class_label*self.batch_size:(class_label*self.batch_size) +self.batch_size, :] = random_class_batch
            labels[class_label*self.batch_size:(class_label*self.batch_size) +self.batch_size, class_label] = 1
        
        random_batch = np.reshape(random_batch, (random_batch.shape[0], random_batch.shape[1], 1))
        
        return(random_batch, labels)

    def cnn_X_epoch(self, epochs):
        self.check_batch_size()
        for epoch in range(1, epochs):
            epoch_cost = 0
            for l in range(1, 50):
               
                train_input, train_labels = self.get_next_batch()
                cost, preds = self.net1.single_core_compute_params(train_input, train_labels, keep_prob=self.keep_prob)
                epoch_cost += cost
                
            print("Epoch ", epoch)
            print("Cost: ", epoch_cost)
            
        
        self.predict_all_cores()
        self.save_predictions_all_cores()
        
        self.save_ims_all_cores()
        
    
    def get_test_labels_single_core(self, core):
        batch_idx = 0
        total_input_vals = self.train_core_pred_sub_labels[core].shape[0]
        batch_size = self.batch_size * self.num_classes

        while (batch_idx+batch_size < total_input_vals):
            
            train_batch = self.train_core_spec[core][batch_idx:batch_idx+batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            preds = self.net1.single_core_predict_labels(train_batch, keep_prob=self.keep_prob)
            preds = one_hot_probability_to_single_label(preds, self.num_classes)
            
            self.train_core_pred_sub_labels[core][batch_idx:batch_idx+batch_size] = preds
            
            batch_idx += batch_size

        train_batch = self.train_core_spec[core][total_input_vals-batch_size:, :]
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        preds = self.net1.single_core_predict_labels(train_batch, keep_prob=self.keep_prob)
        preds = one_hot_probability_to_single_label(preds, self.num_classes)
        self.train_core_pred_sub_labels[core][total_input_vals-batch_size:] = preds
   
    
    def viz_single_core_pred(self, core):

        positions = self.train_positions[core]
        
        xmax = np.max(positions[:, 0])
        xmin = np.min(positions[:, 0])
        ymax = np.max(positions[:, 1])
        ymin = np.min(positions[:, 1])
        image_array = np.zeros((xmax-xmin+1, ymax-ymin+1))
        image_array[:] = 4
        
        cmap = matplotlib.colors.ListedColormap(['black', 'red', 'blue', 'yellow', 'white'])
        bounds = [0, 1, 2, 3, 4,5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        for location in range(0, self.train_core_pred_sub_labels[core].shape[0]):
            label = self.train_core_pred_sub_labels[core][location]
            xloc = self.train_positions[core][location][0]- xmin
            yloc = self.train_positions[core][location][1] - ymin
            
            image_array[xloc, yloc] = label
        
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.grid(True)
        plt.imshow(image_array, interpolation='nearest',cmap=cmap,norm=norm)

        title = "Core Number: " + str(core)  + " Label: " +  self.diagnosis_dict_reverse[self.test_core_true_label[core]]
        plt.title(title)
        plt.colorbar(cmap=cmap,norm=norm,boundaries=bounds,ticks=[0,1,2, 3])
        filename = os.path.join(self.image_out_path, core + '.png')
        
        plt.savefig(filename, pad_inches=0)
        plt.clf()
        
    
    def predict_all_cores(self):
        for core in self.train_data_total.cores_list:
            self.get_test_labels_single_core(core)
            
    
    def get_positions_for_core(self, core):
        positions_filename = os.path.join(self.data_folder, core + '_positions' + '.hdf5')
        with h5py.File(positions_filename, "r") as hf:
            dname = list(hf.keys())[0]
            n1 = hf.get(dname)    
            positions_array = np.copy(n1)

        return(positions_array)

   
    def save_predictions_all_cores(self):
        print("Saving Core Predictions as H5 Files")
        
        for core in self.train_data_total.cores_list:
            prev_labels = self.train_core_pred_sub_labels[core]
            
            labels_filename = os.path.join(self.h5_out_path, core + '_multiclass.hdf5')
            with h5py.File(labels_filename, "w") as f:
                dset = f.create_dataset(core + "multiclass_labels", data=prev_labels, dtype='f')
    
    def save_ims_all_cores(self):
        print("Saving Images . . .")
        for core in self.train_data_total.cores_list:
            self.viz_single_core_pred(core)
    
batch_size = 4 # per 

num_classes = 4
num_epochs=1
lr=.001

MIL = MIL(fc_units = 100, num_classes=num_classes, width1=38,  width2=18, width3=16, filters_layer1=40, 
    filters_layer2=60, filters_layer3=100, batch_size=batch_size, lr=lr, keep_prob=.99)
MIL.cnn_X_epoch(num_epochs)

