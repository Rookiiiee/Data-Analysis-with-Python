import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from IPython.display import clear_output

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Определение слоев сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        # Полносвязные слои
        self.fc1 = nn.Linear(5 * 6 * 6, 100)  # Размер входа после всех сверток и пуллингов - 5 каналов, 6x6 карты активаций
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Размерность х ~ [64, 3, 32, 32]
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Проверка
# Эта ячейка не должна выдавать ошибку.
# Если при исполнении ячейки возникает ошибка, то в вашей реализации нейросети есть баги.
img = torch.Tensor(np.random.random((32, 3, 32, 32)))
model = ConvNet()
out = model(img)

# Проверка устройства сети
# conv1
assert model.conv1.kernel_size == (5, 5), "неверный размер ядра у conv1"
assert model.conv1.in_channels == 3, "неверный размер in_channels у conv1"
assert model.conv1.out_channels == 3, "неверный размер out_channels у conv1"

# pool1
assert model.pool1.kernel_size == (2, 2), "неверный размер ядра у pool1"

# conv2
assert model.conv2.kernel_size == (3, 3), "неверный размер ядра у conv2"
assert model.conv2.in_channels == 3, "неверный размер in_channels у conv2"
assert model.conv2.out_channels == 5, "неверный размер out_channels у conv2"

# pool2
assert model.pool1.kernel_size == (2, 2), "неверный размер ядра у pool2"

# fc1
assert model.fc1.out_features == 100, "неверный размер out_features у fc1"
# fc2
assert model.fc2.out_features == 10, "неверный размер out_features у fc2"
