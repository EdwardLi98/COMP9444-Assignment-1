# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.fc1 = nn.Linear(2, num_hid)
        self.fc2 = nn.Linear(num_hid, 1)

    def forward(self, input):

        polar = torch.stack((input[:,0],input[:,1]), dim=1)
        polar[:,0] = torch.sqrt(input[:,0]*input[:,0] + input[:,1]*input[:,1])
        polar[:,1] = torch.atan2(input[:,1], input[:,0])
        hid_node = F.tanh(self.fc1(polar))
        output = F.sigmoid(self.fc2(hid_node))
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.fc1 = nn.Linear(2, num_hid)
        self.fc2 = nn.Linear(num_hid, num_hid)
        self.fc3 = nn.Linear(num_hid, 1)

    def forward(self, input):
        hid_node1 = F.tanh(self.fc1(input))
        hid_node2 = F.tanh(self.fc2(hid_node1))
        output = F.sigmoid(self.fc3(hid_node2))
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    # INSERT CODE HERE
