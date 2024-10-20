import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision import datasets, transforms

from matplotlib import pyplot as plt

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义网络的各层

        # Conv1: 输入通道数=3，输出通道数=3，卷积核大小=(5,5)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))

        # Pool1: 最大池化，池化核大小=(2,2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # Conv2: 输入通道数=3，输出通道数=5，卷积核大小=(3,3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))

        # Pool2: 最大池化，池化核大小=(2,2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Flatten层，将多维张量展平成一维
        self.flatten = nn.Flatten()

        # 全连接层
        # 计算展平后的特征数量
        # 输入图像大小: 32x32
        # 经过conv1 (kernel=5): (32 - 5 + 1) = 28 -> 28x28
        # 经过pool1 (kernel=2): 28 / 2 = 14 -> 14x14
        # 经过conv2 (kernel=3): (14 - 3 + 1) = 12 -> 12x12
        # 经过pool2 (kernel=2): 12 / 2 = 6 -> 6x6
        # 因此，展平后的特征数量 = 5 * 6 * 6 = 180
        self.fc1 = nn.Linear(in_features=5 * 6 * 6, out_features=100)

        # 全连接层，输出为10个类别
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        # 输入维度 x ~ [batch_size, 3, 32, 32]

        # 通过 Conv1 -> ReLU 激活 -> Pool1
        x = F.relu(self.conv1(x))  # 输出维度: [batch_size, 3, 28, 28]
        x = self.pool1(x)          # 输出维度: [batch_size, 3, 14, 14]

        # 通过 Conv2 -> ReLU 激活 -> Pool2
        x = F.relu(self.conv2(x))  # 输出维度: [batch_size, 5, 12, 12]
        x = self.pool2(x)          # 输出维度: [batch_size, 5, 6, 6]

        # 展平张量
        x = self.flatten(x)        # 输出维度: [batch_size, 180]

        # 通过全连接层
        x = F.relu(self.fc1(x))    # 输出维度: [batch_size, 100]
        x = self.fc2(x)            # 输出维度: [batch_size, 10]

        return x

def create_model():
    return ConvNet()
