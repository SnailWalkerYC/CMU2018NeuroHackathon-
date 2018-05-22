import io
import sys
import os
import re
import numpy as np


# load data into numpy array
data_path = 'data/'
binsize = 10
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
pre = 1000
after = 700

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
    jsPos_change_temp = jsPos_new[:, 1:time_end - 1] - jsPos_new[:, 0:time_end - 2]
    jsPos_change = np.sqrt(np.power(jsPos_change_temp[0], 2) + np.power(jsPos_change_temp[1], 2))
    num_bins = time_end // bin_size
    data1 = np.zeros((data_a1.shape[0], num_bins))
    data2 = np.zeros((data_a2.shape[0], num_bins))
    speed = []
    for i in range(num_bins):
        data1[:, i] = np.sum(data_a1[:, i * bin_size:(i + 1) * bin_size], axis=1)
        data2[:, i] = np.sum(data_a2[:, i * bin_size:(i + 1) * bin_size], axis=1)
        speed.append(np.sum(jsPos_change[i * bin_size:(i + 1) * bin_size]) / bin_size)
    return np.array(speed), np.transpose(data1), np.transpose(data2)


def calculate_cutted_speed(jsPos, data_a1, data_a2, bin_size, reachStart, prev, after):
    # calculates positive velocity amplitude of bin_size (in ms)
    # outputs speed labels and features with each row as a data
    # 1st output in size (number of bins, ) of speed
    # 2nd output in size (number of bins, number of neurons in area1)
    # 3rd output in size (number of bins, number of neurons in area2)
    jsPos_new = jsPos
    data_a1 = np.transpose(data_a1)
    data_a2 = np.transpose(data_a2)
    time_end = jsPos.shape[1]
    data_a1_new = np.zeros((1, 1))
    data_a2_new = np.zeros((1, 1))
    jsPos_new_new = np.zeros((1, 1))
    starts = 0
    for t in reachStart:
        if t > prev:
            start = int(t-prev)
            if t + after < time_end:
                end = int(t+after)
                if len(data_a1_new[0]) < prev:
                    data_a1_new = data_a1[:, start:end]
                    data_a2_new = data_a2[:, start:end]
                    jsPos_new_new = jsPos_new[:, start:end]
                else:
                    data_a1_new = np.concatenate((data_a1_new, data_a1[:, start:end]), axis = 1)
                    data_a2_new = np.concatenate((data_a2_new, data_a2[:, start:end]), axis = 1)
                    jsPos_new_new = np.concatenate((jsPos_new_new, jsPos_new[:, start:end]), axis = 1)
    jsPos_new = jsPos_new_new
    time_end = jsPos_new.shape[1]
    data_a1 = data_a1_new
    data_a2 = data_a2_new
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

def extract_data(tss, time_end):
    tss_new = tss
    num_neurons = tss.shape[0]
    data = np.zeros((num_neurons, time_end))
    for i in range(num_neurons):
        indices = tss_new[i]
        indices = indices[~np.isnan(indices)]
        data[i], bins_edges = np.histogram(indices, bins = time_end, range = (0, time_end))
    return np.transpose(data), bins_edges


# tss2 = np.loadtxt(data_path +'tss2', delimiter=',')*10 #(385, 90631)
# depth2 = np.loadtxt(data_path + 'depth2', delimiter = ',') #(385, )

def load():

    area1_s1 = [depth1 >= Mborder1[0]]
    area2_s1 = [depth1 < Mborder1[0]]
    data1_a1, bins_edges = extract_data(tss1[area1_s1], jsPos1.shape[1])
    data1_a2, bins_edges = extract_data(tss1[area2_s1], jsPos1.shape[1])


    #data = openFile(feature_file)
    #lable = openFile(label_file)
    #data, _  =  extract_data(tss1, jsPos1.shape[1])
    #data = np.asarray(data)
    speed, data1, data2 = \
        calculate_cutted_speed(jsPos1, data1_a1, data1_a2, binsize, reachStart1, pre, after)

    lable = np.array([speed]).T

    data = np.concatenate((data1, data2), axis=1)
    TIME_LENGTH,_ = data.shape

    train_index = list(range(int(3/5*TIME_LENGTH)))
    train_data = data[train_index,:]
    train_label = lable[train_index,:]

    val_index = list(range(int(3/5*TIME_LENGTH), int(4/5*TIME_LENGTH)))
    val_data = data[val_index,:]
    val_lable = lable[val_index,:]


    test_index = list(range(int(4/5*TIME_LENGTH), TIME_LENGTH))
    test_data = data[ test_index,:]
    test_label = data[test_index,:]

    return train_data, val_data, test_data, train_label, val_lable, test_label

if __name__ == '__main__':
    feature = 'feature1_short'
    label_file = 'label1_short'
    data, lable = load()
