#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 18:15:30 2021

@author: stephenschmidt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:37:21 2021

@author: stephenschmidt
"""



import numpy as np
import tensorflow.compat.v1 as v1
v1.disable_eager_execution()
import os
from loadSimData import DirectCNNSimDataset, SimData
from net1_adam import MSInet1
import matplotlib.pyplot as plt
import matplotlib


def single_to_one_hot(labels, num_classes):
        #diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 4}
        # shifted to {'high': 0, 'CA': 1, 'low': 2, 'healthy': 3}
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for hot_class in range(0, num_classes):
        class_locations = np.where(labels == hot_class)
        one_hot_labels[class_locations, hot_class] = 1
    return(one_hot_labels)

def one_hot_probability_to_single_label(pred_labels, prev_labels, num_classes):    
    uncertain_pred_locations = np.where(pred_labels[:, 0]==.5)
    
    argmax_labels = np.argmax(pred_labels, axis=1)
    # Dont update lcoations where prediction output was .5
    if (uncertain_pred_locations[0].shape[0]>0):
        argmax_labels[uncertain_pred_locations[0]] = np.argmax(prev_labels[uncertain_pred_locations])
    return(argmax_labels)


class MIL():
    
    def __init__(self, input_file = 'MICNNSim_Out', fc_units = 50, num_classes=4, width1=38, width2=18, width3=16, filters_layer1=12, 
                 filters_layer2=24, filters_layer3=48, batch_size=4, lr=.001, keep_prob=.8):
        
        self.train_dataset = DirectCNNSimDataset(num_classes=4, train_data_location=input_file)
        
        self.smallROI = SimData()
        self.smallROI.split_cores()
        
        
        self.diagnosis_dict =         {'high': 1, 'CA': 2, 'low': 3, 'healthy': 0}
        self.diagnosis_dict_reverse = {1: 'high', 2: 'CA', 3: 'low', 0:'healthy'}
        
        if num_classes == 2:
            self.diagnosis_dict =         {'high': 1, 'healthy': 0}
            self.diagnosis_dict_reverse = {1: 'high',  0:'healthy'}

        self.batch_size = batch_size
        self.lr = lr
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        
        print("Cores Included For Testing: ", self.smallROI.cores_list)
        
        self.sample_shape = int(self.smallROI.spec[0].shape[0])
        self.net1 = MSInet1(data_shape = self.sample_shape, fc_units=fc_units, num_classes=num_classes, width1=width1, width2=width2, width3=width3, filters_layer1=filters_layer1, filters_layer2=filters_layer2, filters_layer3=filters_layer3, batch_size=batch_size*self.num_classes,lr=lr)
        self.net1.build_graph()
        self.highest_score = .25
        self.average_score = 0
        self.final_score = 0
    

        
    def init_MSI(self):
        
        
        self.test_core_spec = {}
        self.test_core_true_sub_labels = {}
        self.test_core_true_label = {}
        self.test_core_pred_sub_labels = {}
        self.test_core_probability_labels = {}
        self.test_positions = {}
        
        for core in self.smallROI.cores_list:
            core_positions = self.smallROI.core_specific_positions[core]
            self.test_core_spec[core] = self.smallROI.spec[core_positions]
            self.test_core_true_sub_labels[core] = self.smallROI.subtissue_labels[core_positions]
            self.test_core_true_label[core] = int(self.smallROI.tissue_labels[core_positions][0])
            self.test_core_pred_sub_labels[core] = self.smallROI.tissue_labels[core_positions].astype(int)
            self.test_core_probability_labels[core] = np.zeros((self.smallROI.tissue_labels[core_positions].shape[0], self.num_classes))
            self.test_positions[core] = self.smallROI.position[core_positions].astype(int)
    
        
    def get_test_labels_single_core(self, core):
        batch_idx = 0
        labels_changed = 0
        total_input_vals = self.test_core_probability_labels[core].shape[0]
        
        batch_size = self.batch_size * self.num_classes

        while (batch_idx+batch_size < total_input_vals):
            
            train_batch = self.test_core_spec[core][batch_idx:batch_idx+batch_size]
            train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
            
            previous_labels = self.test_core_pred_sub_labels[core][batch_idx:batch_idx+batch_size]

            preds = self.net1.single_core_predict_labels(train_batch, keep_prob=self.keep_prob)
            new_imputed_labels = one_hot_probability_to_single_label(preds, previous_labels, self.num_classes)
            
            self.test_core_probability_labels[core][total_input_vals-batch_size:] = preds
            
            diffs = len(np.where(previous_labels!=new_imputed_labels)[0])
            
            self.test_core_pred_sub_labels[core][batch_idx:batch_idx+batch_size] = new_imputed_labels
            labels_changed+=diffs
            
            batch_idx += batch_size

        train_batch = self.test_core_spec[core][total_input_vals-batch_size:, :]        
        train_batch = np.reshape(train_batch, (train_batch.shape[0], train_batch.shape[1], 1))
        
        train_labels = self.test_core_pred_sub_labels[core][total_input_vals-batch_size:]
        train_labels = single_to_one_hot(train_labels, self.num_classes)
        
        previous_labels = self.test_core_pred_sub_labels[core][total_input_vals-batch_size:]
    
        preds = self.net1.single_core_predict_labels(train_batch, keep_prob=self.keep_prob)
        new_imputed_labels = one_hot_probability_to_single_label(preds, previous_labels, self.num_classes)
        
        
        self.test_core_probability_labels[core][total_input_vals-batch_size:] = preds
        self.test_core_pred_sub_labels[core][total_input_vals-batch_size:] = new_imputed_labels
    
    
    def check_batch_size(self):
        for class_label in range(0, self.num_classes):
            for class_label in range(0, self.num_classes):
                label_spec_size = self.train_dataset.spec_dict[class_label].shape[0]
                if label_spec_size < self.batch_size:
                    print("Class: ", class_label, " Has ", label_spec_size, " Inputs")
                    assert False # Reduce batch Size
                    
        for core in self.smallROI.cores_list:
            labels = self.test_core_true_sub_labels[core].shape[0]
            if labels < self.batch_size*self.num_classes:
                print("Test Core Only Has: ", labels , " inputs")
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

    def cnn_X_epoch(self, x, test_every_x=5):
        self.check_batch_size()
        for epoch in range(1, x):
            epoch_cost = 0
            for l in range(1, 50):
               
                
                train_input, train_labels = self.get_next_batch()
                cost, preds = self.net1.single_core_compute_params(train_input, train_labels, keep_prob=self.keep_prob)
                epoch_cost += cost
                
            if epoch % test_every_x == 0:
                if self.num_classes == 2:
                    self.eval_all_cores2class()
                else:
                    self.eval_all_cores()
                        
            print("Epoch ", epoch)
            print("Cost: ", epoch_cost)
            print('Highest Balanced Accuracy: ', self.highest_score)
                
        
        print('Final Score: ', self.final_score)
        print('Average Score: ', self.average_score/x)
                
                
  
    # Evaulte according to only the given subtissue labels
    def eval_single_core_direct_labels(self, core):
        true_labels = self.test_core_true_sub_labels[core]
        pred_labels = self.test_core_pred_sub_labels[core]
        core_label = self.test_core_true_label[core]
        
        # Case for healthy tissues
        if(core_label == self.diagnosis_dict['healthy']):
            tn = pred_labels.shape[0] - np.sum(pred_labels)
            total_neg = pred_labels.shape[0]
            total_pos = 0
            tp = 0
        
        # Case for non-healthy tissues
        else:
            total_pos = np.where(true_labels == core_label)[0].shape[0]
            total_neg = np.where(true_labels != core_label)[0].shape[0]
            pos_locations = np.where((true_labels == core_label))
            neg_locations = np.where((true_labels != core_label))
            
            expected_core_postive_locations = pred_labels[pos_locations]
            expected_core_negative_locations = pred_labels[neg_locations]
            
            tp = np.where(expected_core_postive_locations == core_label)[0].shape[0]
            tn = np.where(expected_core_negative_locations == self.diagnosis_dict['healthy'])[0].shape[0]
        
        return(tp, tn, total_pos, total_neg, core_label)
    
    def eval_all_cores(self):
        # THis is after enforcing constraints
        #self.diagnosis_dict = {'high': 1, 'CA': 2, 'low': 3, 'healthy': 0}
        tp_high = 0
        tp_CA = 0
        tp_low = 0
        tn_all_cores = 0
        
        total_high = 0
        total_CA = 0
        total_low = 0
        total_neg_all_cores = 0
        for core in self.smallROI.cores_list:
            self.get_test_labels_single_core(core)
            if self.test_core_true_label[core] == self.diagnosis_dict['healthy']:
                continue
            tp, tn, total_pos, total_neg, core_label = self.eval_single_core_direct_labels(core)
            
            if core_label == self.diagnosis_dict['healthy']:
                tn_all_cores += tn
                total_neg_all_cores += total_neg
                
            elif core_label == self.diagnosis_dict['high']:
                tp_high += tp
                total_high += total_pos
                tn_all_cores += tn
                total_neg_all_cores += total_neg
                
            elif core_label == self.diagnosis_dict['CA']:
                tp_CA += tp
                total_CA += total_pos
                tn_all_cores += tn
                total_neg_all_cores += total_neg
                
            elif core_label == self.diagnosis_dict['low']:
                tp_low += tp
                total_low += total_pos
                tn_all_cores += tn
                total_neg_all_cores += total_neg
                
        print('  - Evaluating All Cores - ')
        accuracy = (tp_high + tp_CA + tp_low + tn_all_cores) / (total_high + total_CA + total_low +total_neg_all_cores)
        print('Accuracy: ', accuracy)
        
        if total_neg_all_cores == 0:
            neg_accuracy = -1
        else:
            neg_accuracy = (tn_all_cores / total_neg_all_cores)
        if total_high == 0:
            high_accuracy = -1
        else:
            high_accuracy = (tp_high / total_high)
        if total_CA == 0:
            ca_accuracy = -1
        else:
            ca_accuracy = (tp_CA / total_CA)
        if total_low == 0:
            low_accuracy = -1
        else:
            low_accuracy = (tp_low / total_low)
        
        print('Neg Accuracy: ', neg_accuracy)
        print('High Accuracy: ', high_accuracy)
        print('CA Accuracy: ', ca_accuracy)
        print('Low Accuracy: ', low_accuracy)
        
        balanced_accuracy = (neg_accuracy  * .25) + (high_accuracy*.25) + (ca_accuracy*.25) + (low_accuracy*.25)
        print('Balanced Accuracy: ', balanced_accuracy)
        
        if balanced_accuracy > self.highest_score:
            self.highest_score = balanced_accuracy
            self.save_ims_all_cores()
        
        print('  - Resuming Training  - ')
   
    
    def viz_single_core_pred(self, core):

        positions = self.test_positions[core]
        
        xmax = np.max(positions[:, 0])
        xmin = np.min(positions[:, 0])
        ymax = np.max(positions[:, 1])
        ymin = np.min(positions[:, 1])
        image_array = np.zeros((xmax-xmin+1, ymax-ymin+1))
        image_array[:] = 4
        
        cmap = matplotlib.colors.ListedColormap(['black', 'red', 'blue', 'yellow', 'white'])
        bounds = [0, 1, 2, 3, 4,5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        for location in range(0, self.test_core_pred_sub_labels[core].shape[0]):
            label = self.test_core_pred_sub_labels[core][location]
            xloc = self.test_positions[core][location][0]- xmin
            yloc = self.test_positions[core][location][1] - ymin
            
            image_array[xloc, yloc] = label
        
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.grid(True)
        plt.imshow(image_array, interpolation='nearest',cmap=cmap,norm=norm)

        title = "Core Number: " + str(core)  + " Label: " +  self.diagnosis_dict_reverse[self.test_core_true_label[core]]
        #print(title)
        plt.title(title)
        plt.colorbar(cmap=cmap,norm=norm,boundaries=bounds,ticks=[0,1,2, 3])
        filename = 'SimImages/Pred'+ str(int(core)) + '.png'
        #print(filename)
        
        plt.savefig(filename, pad_inches=0)
        plt.clf()
    
    def save_ims_all_cores(self):
        print("Saving Images . . .")
        for core in self.smallROI.cores_list:
            self.viz_single_core_pred(core)
    
batch_size = 3 # per 
small_train = False
input_file = 'Sim88_2_out'

test_every_x = 5
num_epochs=100000
lr=.001

MIL = MIL(input_file = input_file, fc_units = 100, num_classes=4, width1=38,  width2=18, width3=16, filters_layer1=40, 
    filters_layer2=60, filters_layer3=100, batch_size=batch_size, lr=lr, keep_prob=.99)
MIL.init_MSI()
MIL.cnn_X_epoch(num_epochs,test_every_x=test_every_x)



