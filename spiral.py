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
        self.hid_node1 = F.tanh(self.fc1(polar))
        output = F.sigmoid(self.fc2(self.hid_node1))
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.fc1 = nn.Linear(2, num_hid)
        self.fc2 = nn.Linear(num_hid, num_hid)
        self.fc3 = nn.Linear(num_hid, 1)

    def forward(self, input):
        self.hid_node1 = F.tanh(self.fc1(input))
        self.hid_node2 = F.tanh(self.fc2(self.hid_node1))
        output = F.sigmoid(self.fc3(self.hid_node2))
        return output

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): 
            net.eval()        
            output = net(grid)
            if layer == 1:
                pred = (net.hid_node1[:, node]>=0).float()
            elif layer == 2:
                pred = (net.hid_node2[:, node]>=0).float()

            # plot function computed by model
            plt.clf()
            plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')