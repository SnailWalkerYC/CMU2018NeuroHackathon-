#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import sys
import copy
import argparse
import matplotlib.pyplot as plt
import random
import io
import os
import re
import math

def get_batch(data, label ):
   index = np.random.permutation(5000-500)[0:BATCH_SIZE]
   #global BATCH_START, TIME_STEPS
   # xs shape (50batch, 20steps)
   #xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
   #seq = np.sin(xs)
   #res = np.cos(xs)
   #BATCH_START += TIME_STEPS
   # plt.plot(xs[0, :], res[0, :], ‘r’, xs[0, :], seq[0, :], ‘b--’)
   # plt.show()
   # returned seq, res and xs: shape (batch, step, input)
   #in_data, out_data = data_load
   x = np.empty([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
   y = np.empty([BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE])
   for i in range(BATCH_SIZE):
       x[i, :, :] = data[index[i]:index[i]+TIME_STEPS, :]
       y[i, :, :] = label[index[i]:index[i]+TIME_STEPS, :]
   return x, y

def get_batch(data, label ):
    index = np.random.permutation(5000-500)[0:BATCH_SIZE]
    #global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    #xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    #seq = np.sin(xs)
    #res = np.cos(xs)
    #BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    #in_data, out_data = data_load
    x = np.empty([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
    y = np.empty([BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE])
    for i in range(BATCH_SIZE):
        x[i, :, :] = data[index[i]:index[i]+TIME_STEPS, :]
        y[i, :, :] = label[index[i]:index[i]+TIME_STEPS, :]
    return x, y

def openFile(file_path):
    data = []
    with io.open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            nums = line.split(',')
            if len(nums)>0:
             tmp = []
             for num in nums:
                 tmp.append(float(num))
                 # print(num)
             data.append(tmp)
    print(data[0])
    return data
#    return
def load(feature_file, label_file):
    data = openFile(feature_file)
    lable = openFile(label_file)
    return np.asarray(data), np.asarray(lable)

import numpy as np

# load data into numpy array
data_path = 'data/'

tss1 = np.loadtxt(data_path +'tss1', delimiter=',')*10 #(255, 159086)
# depth is a vector of length(neuron), measured in mm. Larger numbers are deeper. 
depth1 = np.loadtxt(data_path + 'depth1', delimiter = ',') #(255, )
reaches1 = np.loadtxt(data_path + 'reaches1', delimiter = ',') #(163, 1500)
Mborder1 = np.loadtxt(data_path + 'Mborder1', delimiter = ',') #(2, )
# jsPos is joystick behavior file.  Row 1 is X position,  Row 2 is Y position.  
jsPos1 = np.loadtxt(data_path + 'jsPos1', delimiter = ',') # (2, 2325146)
reachStart1 = np.loadtxt(data_path + 'reachStart1', delimiter = ',') #(164,)
rewTime1 = np.loadtxt(data_path + 'rewTime1', delimiter = ',') #(217, )
reaches1 = np.loadtxt(data_path + 'reaches1', delimiter = ',') #(163, 1500)
ampVel1 = np.loadtxt(data_path + 'ampVel1', delimiter = ',') # (163, 2)

# tss2 = np.loadtxt(data_path +'tss2', delimiter=',')*10 #(385, 90631)
# depth2 = np.loadtxt(data_path + 'depth2', delimiter = ',') #(385, )

def extract_data(tss, time_end):
    tss_new = tss
    num_neurons = tss.shape[0]
    data = np.zeros((num_neurons, time_end))
    for i in range(num_neurons):
        indices = tss_new[i]
        indices = indices[~np.isnan(indices)]
        data[i], bins_edges = np.histogram(indices, bins = time_end, range = (0, time_end))
    return np.transpose(data), bins_edges

area1_s1 = [depth1 >= Mborder1[0]]
area2_s1 = [depth1 < Mborder1[0]]
data1_a1, bins_edges= extract_data(tss1[area1_s1], jsPos1.shape[1])
data1_a2, bins_edges = extract_data(tss1[area2_s1], jsPos1.shape[1])

np.savetxt('feature_s1_a1', data1_a1, fmt = '%d', delimiter = ',')
np.savetxt('feature_s1_a2', data1_a2, fmt = '%d', delimiter = ',')
np.savetxt('label_s1', np.transpose(jsPos1), fmt = '%.4f', delimiter = ',')


def calculate_speed(jsPos, data_a1, data_a2, bin_size):
    # calculates velocity amplitude of bin_size (in ms)
    # outputs speed labels and features with each row as a data  
    # 1st output in size (number of bins, ) of speed
    # 2nd output in size (number of bins, number of neurons in area1)
    # 3rd output in size (number of bins, number of neurons in area2)
    
    jsPos_new = jsPos
    data_a1 = np.transpose(data_a1)
    data_a2 = np.transpose(data_a2)
    time_end = jsPos.shape[1]
    jsPos_change_temp = jsPos_new[:, 1:time_end-1] - jsPos_new[:, 0:time_end-2]
    jsPos_change = np.sqrt(np.power(jsPos_change_temp[0],2) + np.power(jsPos_change_temp[1],2))
    num_bins = time_end//bin_size
    data1 = np.zeros((data_a1.shape[0], num_bins))
    data2 = np.zeros((data_a2.shape[0], num_bins))
    speed = []
    for i in range(num_bins):
        data1[:, i] = np.sum(data_a1[:, i*bin_size:(i+1)*bin_size], axis = 1)
        data2[:, i] = np.sum(data_a2[:, i*bin_size:(i+1)*bin_size], axis = 1)
        speed.append(np.sum(jsPos_change[i*bin_size:(i+1)*bin_size])/bin_size)
    return np.array(speed), np.transpose(data1), np.transpose(data2)

def avaliable_time( after, data):
    ava_list = []
    startime = np.arange(100, data.shape[0] - 70, 70)
    for i in startime:
        tmp = list(range(i, i + after))
        ava_list.append(tmp)
    return ava_list

if __name__ == '__main__':
    feature = 'feature1_short'
    label_file = 'label1_short'
    data, lable = load(feature, label_file)       