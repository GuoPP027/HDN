#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   backbones.py
@Time    :   2021/11/24 10:25:13
@Author  :   Guo Peng
@Version :   1.0
@Contact :   guopengeic@163.com
'''

from torch import nn
from torchvision.models import resnet
import torch.utils.model_zoo as model_zoo


# weight repository
model_urls = {
    'resnet18': ''
}

class ResNet(nn.Module):
    def __init__(self, name = "resnet18", pretrain=True):
        super().__init__()
        if name == "resnet18":
            base_net = resnet.resnet18(pretrained=False)
        else:
            print("base model is not support")
        
        if pretrain:
            print("load the {} weight from ./cache".format(name))
            base_net.load_state_dict(model_zoo.load_url(model_urls[name], model_dir="./cache"))
        self.model = base_net

    def forword(self, x):
        return self.model(x)
