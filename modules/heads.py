#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   heads.py
@Time    :   2021/11/24 10:25:07
@Author  :   Guo Peng
@Version :   1.0
@Contact :   guopengeic@163.com
'''

import numpy as np

class DN:
    def __init__(self, input_dim, y_neuron_num, y_top_k, z_neuron_num):
        self.x_neuron_num = 1
        self.input_dim = input_dim
        input_dim = np.array([input_dim]).reshape(-1)
        for item in input_dim:
            self.x_neuron_num = self.x_neuron_num * item
        self.y_neuron_num = y_neuron_num
        self.z_area_num = len(z_neuron_num)
        self.z_neuron_num = z_neuron_num
        self.y_top_k = y_top_k
        self.dn_create()

    def dn_create(self):
        # some super parameters
        self.y_bottom_up_percent = 1 / 2
        self.y_top_down_percent = 1 / 2

        # responses
        self.x_response = np.zeros((1, self.x_neuron_num))

        self.y_bottom_up_response = np.zeros((1, self.y_neuron_num))
        self.y_top_down_response = np.zeros((self.z_area_num, self.y_neuron_num))
        self.y_pre_response = np.zeros((1, self.y_neuron_num))

        self.y_response = np.zeros((1, self.y_neuron_num))
        self.z_response = []
        for i in range(self.z_area_num):
            self.z_response.append(np.zeros((1, self.z_neuron_num[i])))

        # weights
        self.y_bottom_up_weight = np.random.random_sample((self.y_neuron_num, self.x_neuron_num))
        self.y_top_down_weight = []
        for i in range(self.z_area_num):
            self.y_top_down_weight.append(np.random.random_sample((self.y_neuron_num, self.z_neuron_num[i])))

        # firing the initial neuron
        self.y_threshold = np.zeros((1, self.y_neuron_num))

        # age and flag
        self.y_lsn_flag = np.zeros((1, self.y_neuron_num))
        self.y_firing_age = np.zeros((1, self.y_neuron_num))
        self.set_flag = np.zeros((1, self.y_neuron_num))

        # z weights
        self.z_bottom_up_weight = []
        self.z_firing_age = []
        for i in range(self.z_area_num):
            self.z_bottom_up_weight.append(np.zeros((self.y_neuron_num, self.z_neuron_num[i])))  # ????
            self.z_firing_age.append(np.zeros((1, self.z_neuron_num[i])))

    def preprocess(self, x):
        x = x - np.mean(x)
        max_val = np.max(x)
        min_val = np.min(x)
        if (max_val - min_val) != 0:
            x = (x - min_val) / (max_val - min_val)
        return x

    def compute_response(self,input_vec, weight_vec):
        neuron_num, input_dim = weight_vec.shape  # Y*X
        temp = np.tile(input_vec, (neuron_num, 1))
        temp = self.normalize(temp)
        weight_vec = self.normalize(weight_vec)
        # for i in range(neuron_num):
        #     result[0][i] = np.dot(temp[i].reshape(1, -1), weight_vec[i].reshape(-1, 1))[0, 0]
        result = np.sum(temp * weight_vec, axis=1)
        return result  # 1 x 25

    def normalize(self, input):
        _, input_dim = input.shape
        norm = np.sqrt(np.sum(input * input, axis=1))
        norm[norm==0] = 1
        result = input / np.tile(norm.reshape(-1, 1), (1, input_dim))
        return result

    def get_learning_rate(self, firing_age):
        lr = 1.0 / (firing_age + 1.0)
        if lr < 1.0 / 50.0:
            lr = 1.0 / 50.0
        return lr

    def mean(self, input_vec):
        input_vec = input_vec.reshape(1, -1)
        _, lenth = input_vec.shape
        use_lenth = 0
        mean = 0
        for i in range(lenth):
            if input_vec[0][i] > 0:
                use_lenth += 1
                mean += input_vec[0][i]

        if use_lenth == 0:
            use_lenth = 1
        return mean / use_lenth

    def dn_learn(self, training_image, true_z):
        self.x_response = training_image.reshape(1, -1)
        for i in range(self.z_area_num):
            self.z_response[i] = np.zeros(self.z_response[i].shape)
            self.z_response[i][0, true_z[i]] = 1
        self.x_response = self.preprocess(self.x_response)

        # compute response
        self.y_bottom_up_response = self.compute_response(self.x_response,
                                                          self.y_bottom_up_weight)

        for i in range(self.z_area_num):
            self.y_top_down_response[i] = self.compute_response(self.z_response[i],
                                                                self.y_top_down_weight[i])
        # top-down + bottom-up response
        self.y_pre_response = (self.y_bottom_up_percent * self.y_bottom_up_response +
                               self.y_top_down_percent * np.mean(self.y_top_down_response, axis=0).reshape(1,
                                                                                                           -1)) / (
                                      self.y_bottom_up_percent + self.y_top_down_percent)  # mean or with weight

        max_response, max_index = self.top_k_competition(True)

        self.hebbian_learning()
        self.updateInhibit(max_index, max_response)
        return max_response, max_index

    def updateInhibit(self, max_index, max_response):
        # inhibit  Y response
        lr = self.get_learning_rate(self.y_firing_age[0][max_index]-1)
        self.y_threshold[0][max_index] = lr * max_response + (1-lr)*self.y_threshold[0][max_index]

    def top_k_competition(self, flag):
        if flag:
            self.y_response = np.zeros(self.y_response.shape)
            seq_high = np.argsort(-self.y_pre_response)
            max_response = self.y_pre_response[0][seq_high[0][0]]
            if max_response > self.y_threshold[0][seq_high[0][0]]:
                self.y_response[0][seq_high[0][0]] = 1
                self.set_flag[0][seq_high[0][0]] = 1
                return max_response, seq_high[0][0]

            if self.set_flag[0][seq_high[0][0]] < 1:
                self.y_response[0][seq_high[0][0]] = 1
                self.set_flag[0][seq_high[0][0]] = 1
                return max_response, seq_high[0][0]
            else:
                _, lenth = seq_high.shape
                for i in range(lenth):
                    if self.set_flag[0][seq_high[0][i]] <1:
                        self.y_response[0][seq_high[0][i]] = 1
                        self.set_flag[0][seq_high[0][i]] = 1
                        return max_response, seq_high[0][i]
            self.y_response[0][seq_high[0][0]] = 1
            return max_response, seq_high[0][0]
        else:
            self.y_response = np.zeros(self.y_response.shape)
            seq_high = np.argsort(-self.y_pre_response)
            max_response = self.y_pre_response[0][seq_high[0][self.y_top_k - 1]]
            self.y_response[0][seq_high[0][self.y_top_k - 1]] = 1
            return max_response, seq_high[0][0]

    def hebbian_learning(self):
        for i in range(self.y_neuron_num):
            if self.y_response[0, i] == 1:  # firing neuron, currently set response to 1
                if self.y_lsn_flag[0, i] == 0:
                    self.y_lsn_flag[0, i] = 1
                    self.y_firing_age[0, i] = 0
                lr = self.get_learning_rate(self.y_firing_age[0, i])  # learning rate
                # self.y_bottom_up_weight[i] = normalize(self.y_bottom_up_weight[i], mask)

                self.y_bottom_up_weight[i] = (1 - lr) * self.y_bottom_up_weight[i] + lr * self.x_response

                # top-down weight and synapse factor
                for j in range(self.z_area_num):
                    # self.z_response[j] = self.normalize(self.z_response[j], np.ones(self.z_response[j].shape))
                    self.y_top_down_weight[j][i] = (1 - lr) * self.y_top_down_weight[j][i] + lr * self.z_response[
                        j]
                self.y_firing_age[0, i] = self.y_firing_age[0, i] + 1

        # z neuron learning
        for area_idx in range(self.z_area_num):
            for i in range(self.z_neuron_num[area_idx]):
                if self.z_response[area_idx][0, i] == 1:
                    lr = self.get_learning_rate(self.z_firing_age[area_idx][0, i])
                    self.z_bottom_up_weight[area_idx][:, i] = (1 - lr) * self.z_bottom_up_weight[area_idx][:,
                                                                         i] + lr * self.y_response.reshape(-1)

                    self.z_firing_age[area_idx][0, i] = self.z_firing_age[area_idx][0, i] + 1