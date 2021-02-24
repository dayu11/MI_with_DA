import torch
import torch.nn as nn
import numpy as np
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ConvNet(nn.Module):

    def __init__(self, channels):
        super(ConvNet, self).__init__()

        self.conv1 = conv3x3(3, channels)
        self.conv2 = conv3x3(channels, channels)

        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(channels*15*15, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, rnd_forward=False, resume=False):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.tanh(x)

        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)

        return x



def convnet(channels):

    model = ConvNet(channels)
    return model