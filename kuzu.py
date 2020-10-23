# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        input_tensor = x.view(-1, 28*28)
        output = self.fc(input_tensor)
        return F.log_softmax(output, dim=1) 

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.fc1 = nn.Linear(28*28, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        input_tensor = x.view(-1, 28*28)
        hid_layer = F.tanh(self.fc1(input_tensor))
        output = self.fc2(hid_layer)
        return F.log_softmax(output, dim=1)

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3) 
        self.fc1 = nn.Linear(24*24*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 24*24*64)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        return x 
