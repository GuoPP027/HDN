#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   HDN.py
@Time    :   2021/11/29 13:41:32
@Author  :   Guo Peng
@Version :   1.0
@Contact :   guopengeic@163.com
'''

import torch
from torch import nn
import module
import os
from tqdm import tqdm
import numpy as np

class JointNet:
    def __init__(self,y_neuron_num, out_dim, device):
        # 获取机器状态
        self.device = device
        # 获取输入输出信息
        self.dim = out_dim
        self.y_top_k = 1
        self.y_neuron_num = y_neuron_num
        # 搭建模型
        self.resNet = module.ResNet(self.dim)
        # self.resNet = torch.load(r'D:\Cifar10\FCSM\Log\net_005.pth')
        self.backBone = self.resNet.resLayers
        # 这里应该自动获取backbone输出的尺寸
        self.input_dim = [64, 8, 8]
        # 这里应该使用细胞分裂机制
        self.developNet = module.DN(self.input_dim, self.y_neuron_num, self.y_top_k, [self.dim])

    # 阶段1，为BackBone的训练阶段，
    # 使用平均池化+一层FC的分类器,交叉损失和SGD优化
    # 设计一种自动训练机制，保留测试准确率最高的模型
    def train_stage_1(self, EPOCHS,trainloader, testloader, criterion, optimizer, net_save_file = './Log'):
        self.resNet.to(self.device)
        outf = net_save_file
        if not os.path.exists(outf):
            os.makedirs(outf)
        bestacc = 0
        bestpath = ''
        print('----- Train Stage 1 Start -----')
        with open(outf+'/acc.txt', 'w') as f:
            for epoch in tqdm(range(EPOCHS)):
                print('\nEpoch: %d' % (epoch + 1))
                self.train(self.resNet, trainloader, optimizer, criterion)
                correct = self.test(self.resNet, testloader)
                print('测试分类准确率为：%.3f%%' % (correct))
                # 将每次测试结果实时写入acc.txt文件中
                f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, correct))
                f.write('\n')
                f.flush()
                if correct > bestacc:
                    # 删除当前最好的模型，保存最好的模型
                    if not bestacc:
                        torch.save(self.resNet, '%s/net_%03d.pth' % (outf, epoch + 1))
                        bestpath = '%s/net_%03d.pth' % (outf, epoch + 1)
                    os.remove(bestpath)
                    torch.save(self.resNet, '%s/net_%03d.pth' % (outf, epoch + 1))
                    bestpath = '%s/net_%03d.pth' % (outf, epoch + 1)
                    bestacc = correct
        print('----- Train Stage 1 Finished -----')
        return bestacc, bestpath

    # 阶段2，冻结backbone的参数，使用其提取的512*4*4特征输入DN的X层
    def train_stage_2(self, EPOCHS, trainloader, testloader, net_save_path='./Log'):
        self.resNet.to(self.device)
        # 联合训练时的dataloader的batch-size是1
        print("-----Train Stage 2 Start-----------")
        outf = net_save_path
        if not os.path.exists(outf):
            os.makedirs(outf)
        bestacc = 0
        self.resNet.eval()

        with open(outf+'/accDN.txt', 'w') as f2:
            for epoch in range(EPOCHS):
                for i, data in tqdm(enumerate(trainloader, 0)):
                    # 准备数据
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    feature = self.backBone(inputs)
                    # feature = torch.squeeze(feature)
                    feature = feature.detach().numpy()
                    self.developNet.dn_learn(feature, labels)
                    break
                print('\nEpoch: %d' % (epoch + 1))
                correct = 0
                total = 0
                for i, data in enumerate(testloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    feature = self.backBone(inputs)
                    feature = feature.detach().numpy()
                    output = self.developNet.dn_test(feature)
                    # 这里需要处理一下
                    predicted = [np.argmax(output[0])]
                    total += 1
                    labels = labels.detach().numpy().tolist()
                    if predicted == labels:
                        correct += 1
                    break
                accuracy = 100 * correct / total
                print('测试分类准确率为：%.3f%%' % (accuracy))
                # 将每次测试结果实时写入acc.txt文件中
                f2.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, accuracy))
                f2.write('\n')
                f2.flush()
                if correct > bestacc:
                    # 删除当前最好的模型，保存最好的模型
                    if not bestacc:
                        # save the current one
                        pass
                    # remove the old one
                    # save the current one
                    bestacc = correct
        print('----- Train Stage 2 Finished -----')


    # 循环训练，设计损失，使用DN的X->Y权重与BackBone的输出之间的距离
    # 去优化BackBone,基于优化的BackBone去优化DN。
    # 可以先使用一次训练去验证可行性，不着急加循环训练
    class DN2Classifar(nn.Module):
        def __init__(self, x2y, y2z):
            super(DN2Classifar, self).__init__()
            layer12 = torch.Tensor(x2y)
            layer23 = torch.Tensor(y2z)
            #正则化
            self.Liner1 = nn.Linear(64*8*8, 5000)
            self.Liner1.weight = nn.Parameter(layer12)
            self.Liner2 = nn.Linear(5000, 10)
            self.Liner2.weight = nn.Parameter(layer23)


        def forward(self, input):
            #x要归一化和正则化
            x = self.Liner1(input)
            max_dim = torch.argmax(x)
            output = x.new(x.size())
            output[max_dim] = 1
            x = self.Liner2(output)
            return x

        def backward(self):
            #梯度为0
            pass

    def train_stage_3(self, out_epochs, epochs,trLoader1, teLoader1, trLoader2, teLoader2, criterion, optimizer,net_save_path='./Log'):
        # 联合训练时的dataloader的batch-size是1
        print("-----Train Stage 3 Start-----------")
        outf = net_save_path
        if not os.path.exists(outf):
            os.makedirs(outf)
        bestacc = 0
        with open(outf+'/accDN.txt', 'w') as f3:
            for out_epoch in range(out_epochs):
                # 这里对net做一些设置
                self.resNet.classifar = self.DN2Classifar()  #需要一些参数
                self.resNet.to(self.device)
                self.resNet.train()
                # 这里的优化函数是一个字典
                self.train_stage_1(epochs,trLoader1, teLoader1, criterion, optimizer)
                # 再做一些设置
                self.resNet.to('cpu')
                self.resNet.eval()
                self.train_stage_2(1, trLoader2, teLoader2)
        print('----- Train Stage 3 Finished -----')



    # 为联合模型设计测试函数，基本上是DN的测试函数
    def test_jointNet(self):
        pass

    # 保存模型
    def save_jiontNet(self):
        pass

    # 加载模型
    def load_jiontNet(self):
        pass

    def train(self, net, dataloader, optimizer, criterion):
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(dataloader, 0):

            # 准备数据
            length = len(dataloader)
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('Loss: %.03f | Acc: %.3f%% ' % (sum_loss / (i + 1), 100. * correct / total))
            break

    def test(self, model, testloader):
        print('------ Test Start -----')
        with torch.no_grad():
            correct = 0
            total = 0
            for test_x, test_y in testloader:
                model.eval()
                images, labels = test_x.to(self.device), test_y.to(self.device)
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                break
        accuracy = 100 * correct / total
        return accuracy