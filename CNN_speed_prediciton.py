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

training_process = []

def open_file(file_path):
    data = []
    with io.open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip() 
            nums = line.split(',')
            if len(nums)>0:
                tmp = []
                for num in nums:
                    tmp.append(float(num))
                data.append(tmp)           
    return data

def calculate_speed(jsPos, data_a1, bin_size):
    # calculates velocity amplitude of bin_size (in ms)
    # outputs speed labels and features with each row as a data  
    # 1st output in size (number of bins, ) of speed
    # 2nd output in size (number of bins, number of neurons in area1)
    # 3rd output in size (number of bins, number of neurons in area2)
    
    jsPos_new = jsPos
    data_a1 = np.transpose(data_a1)
    # data_a2 = np.transpose(data_a2)
    time_end = jsPos.shape[1]
    jsPos_change_temp = jsPos_new[:, 1:time_end-1] - jsPos_new[:, 0:time_end-2]
    jsPos_change = np.sqrt(np.power(jsPos_change_temp[0],2) + np.power(jsPos_change_temp[1],2))
    num_bins = time_end//bin_size
    data1 = np.zeros((data_a1.shape[0], num_bins))
    # data2 = np.zeros((data_a2.shape[0], num_bins))
    speed = []
    for i in range(num_bins):
        data1[:, i] = np.sum(data_a1[:, i*bin_size:(i+1)*bin_size], axis = 1)
        # data2[:, i] = np.sum(data_a2[:, i*bin_size:(i+1)*bin_size], axis = 1)
        speed.append(np.sum(jsPos_change[i*bin_size:(i+1)*bin_size])/bin_size)
    return np.array(speed), np.transpose(data1) #, np.transpose(data2)

class Network():
    def __init__(self):
        tf.reset_default_graph()
        self.learning_rate = 0.0005 # 0.001
        self.hidden_layer_0_size = 24
        self.hidden_layer_1_size = 25
        self.hidden_layer_2_size = 32
        self.hidden_layer_3_size = 26
        self.input_dim = 255
        self.output_dim = 1
        self.x_input = tf.placeholder(tf.float32, [None,self.input_dim])
        self.y_label = tf.placeholder(tf.float32, [None, self.output_dim])
        self.hidden_layer_0, self.W0, self.B0 = self.add_hidden_layer(self.x_input,
                                                                      self.input_dim,
                                                                      self.hidden_layer_0_size,
                                                                      tf.nn.relu)
        self.hidden_layer_1, self.W1, self.B1 = self.add_hidden_layer(self.hidden_layer_0,
                                                                      self.hidden_layer_0_size,
                                                                      self.hidden_layer_1_size,
                                                                      tf.nn.relu)
        self.hidden_layer_2, self.W2, self.B2 = self.add_hidden_layer(self.hidden_layer_1,
                                                                      self.hidden_layer_1_size,
                                                                      self.hidden_layer_2_size,
                                                                      tf.nn.relu)
        self.y_pred, self.W3, self.B3 = self.add_hidden_layer(self.hidden_layer_2,
                                                                  self.hidden_layer_2_size,
                                                                  self.output_dim,
                                                                  None)
        self.mse = tf.square(tf.reduce_sum(self.y_pred - self.y_label))
        self.loss = tf.reduce_mean(self.mse)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.trainig_process = self.optimizer.minimize(self.loss)
        self.inital = tf.global_variables_initializer()
        self.old_weights = [self.W0, self.B0, self.W1, self.B1, self.W2, self.B2]

    def add_hidden_layer(self, x_input, input_size, output_size, activation_function):
        W = tf.Variable(tf.random_normal([input_size, output_size],
                                         mean=0.0,
                                         stddev = 1.0))
        B = tf.Variable(tf.ones(output_size))
        WxB = tf.matmul(x_input, W) + B
        if activation_function:
            Output = activation_function(WxB)
        else:
            Output = WxB
        return Output,W,B

    def save_model_weights(self, suffix):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, suffix)
        return save_path

    def load_model_weights(self,weight_file):
        saver = tf.train.Saver()
        saver.restore(self.sess, weight_file)

class Replay_Memory():
    def __init__(self,features,labels): # 10000
        self.features = features
        self.labels = labels
        self.batch_size = 32
        # store training data.
        # self.replay_memory = open_file(feature_path)
        # self.labels = open_file(label_path)

    def sample_batch(self, batch_size=32):
        self.batch_size = batch_size
        sample_idx = np.random.permutation(len(self.features))[0:batch_size]
        sample_feature = []
        sample_label = []
        for idx in sample_idx:
            sample_feature.append(self.features[idx])
            sample_label.append(self.labels[idx])
        return sample_feature, sample_label 

class DQN_Agent():
    def __init__(self,features,labels):
        self.agent = Network()
        self.memory = Replay_Memory(features, labels)
        self.num_training_epoch = 10
        self.max_iteration = 50000 #10000
        self.update_frequency = 1
        self.agent.sess = tf.Session()
        self.agent.sess.run(self.agent.inital)
        self.update_frequency = 1000 #1000
        self.batch_size = self.memory.batch_size

    def get_batch(self, ind, features, labels):
        sample_feature = []
        sample_label = []
        for idx in range(self.batch_size):
            sample_feature.append(self.features[idx+ind])
            sample_label.append(self.labels[idx+ind])
        return sample_feature, sample_label    

    def meanSquareError(self,nums1, nums2, size=32):
        error = 0.0
        print('-- size of nums1 is %d and nums2 is %d' % (len(nums1), len(nums2)))
        # print(nums1)
        assert(len(nums1)==len(nums2))
        nums_len = len(nums2)
        for ind in range(nums_len):
            error += abs(nums1[ind]-nums2[ind])*abs(nums1[ind]-nums2[ind])
        return math.sqrt(error*1.0/size)         

    def train(self):
        # old_weights = self.agent.sess.run(self.agent.old_weights)
        images = []
        cnt = 0;
        for epoch in range(self.num_training_epoch):
            for step in range(self.max_iteration):
                sample_feature, sample_label = self.memory.sample_batch()
                # print('-- size of sample features %d' % (len(sample_feature)))
                error = 0.0
                if sample_label:
                    feed_dict = {
                        self.agent.x_input: sample_feature,
                        self.agent.y_label: sample_label
                    }
                    _,pred_label = self.agent.sess.run([self.agent.trainig_process,self.agent.y_pred],
                                        feed_dict=feed_dict)
                    error = self.meanSquareError(pred_label,sample_label)
                    
                    # print(error)
                    print('-- The %d epoch %d steps training error is %f\n' % (epoch, step, error))
                if cnt%self.update_frequency == 0:
                    images.append(error)
                    old_weights = self.agent.sess.run(self.agent.old_weights)
                    self.agent.save_model_weights('./Model_' + str(epoch) + '_' + str(step))
                cnt += 1   
        global training_process
        training_process = images         
        plt.plot(images)
        plt.xlabel('Time')
        plt.ylabel('Mean Squre Error with testing step')
        plt.title('Training Process mse vs time')
        plt.savefig('traing_stage_error.png')

    def drawBothImages(self,images):
        # plt.figure() 
        plt.gca().set_color_cycle(['blue', 'orange'])
        plt.plot(training_process)
        plt.plot(images)
        
        # print(training_process)
        # print(images)
        plt.legend(['train curve', 'test curve'], loc='upper right')
        plt.xlabel('Time')
        plt.ylabel('Mean Squre Error with testing step')
        plt.title('Training and test Process mse vs time')
        plt.savefig('both_stage_error.png')
        # plt.show() 
    
    def loadAllData(self,feature_path, label_path):
        features = open_file(feature_path)
        labels = open_file(label_path)
        return features, labels

    def test(self, test_feature, test_label):
        # test_feature, test_label = self.loadAllData(feature_path, label_path)
        print('-- Test process for different model --')
        images = []
        cnt = 0
        for epoch in range(self.num_training_epoch):
            for step in range(self.max_iteration):
                if cnt%self.update_frequency == 0:
                    self.agent.load_model_weights('./Model_' + str(epoch) + '_' + str(step))
                    cnt2 = 0
                    error_tmp = 0.0
                    lens_arr = len(test_feature) 
                    # for idx in range(0, 10000, 32):
                    #    sample_feature, sample_label = self.get_batch(idx, test_feature, test_label)
                    # print('-- Size of sample feature and label is %d and %d ' % (len(sample_feature), len(sample_label)))
                    test_pred = self.agent.sess.run(self.agent.y_pred,
                                            feed_dict={self.agent.x_input:
                                                        test_feature})
                    error_tmp = self.meanSquareError(test_pred, test_label, lens_arr)    
                    images.append(error_tmp)
                    print('-- Test process %d epoch %d steps training error is %f\n' % (epoch, step, error_tmp))
                cnt += 1          
        plt.plot(images)
        plt.xlabel('Time')
        plt.ylabel('Mean Squre Error with testing step')
        plt.title('Test Process mse vs time')
        plt.savefig('test_stage_error.png')
        # plt.show()
        
        self.drawBothImages(images)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    return parser.parse_args()


def load_all_data(path1, path2):
    labels = open_file(path2)
    feature1 = open_file(path1)
#    feature2 = open_file(path2)
    lens = len(feature1)
    '''
    for ind in range(lens):
        feature1[ind].extend(feature2[ind])
    '''    
    num_train_sample = int(7.0/10.0*lens) 
    all_idx = np.random.permutation(lens)
    train_idx = all_idx[0:int(num_train_sample)]
    test_idx = all_idx[num_train_sample+1:lens-1]
    train_feature = []
    train_label = []
    test_feature = []
    test_label = []
    for ind in train_idx:
        train_feature.append(feature1[ind])
        train_label.append(labels[ind])
    cnt = 0    
    for ind in test_idx:
        if cnt>=10000:
            break
        cnt+=1    
        test_feature.append(feature1[ind])
        test_label.append(labels[ind])            
    return train_feature, train_label, test_feature, test_label    

def main(paths):
    training_feature, training_label, test_feature, test_label = load_all_data(paths[1],paths[2])
    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    # gpu_ops = tf.GPUOptions(allow_growth=True)
    # config = tf.ConfigProto(gpu_options=gpu_ops)
    # sess = tf.Session(config=config)
    agent = DQN_Agent(training_feature, training_label)

    agent.train()
    # save_path = agent.agent.save_model_weights('./weights_' + 'test')
    # print(save_path)
    # agent.agent.load_model_weights(save_path)
    agent.test(test_feature,test_label)

if __name__ == '__main__':
    main(sys.argv)

