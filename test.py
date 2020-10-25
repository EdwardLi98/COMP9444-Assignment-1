import torch
import torch.nn as nn

xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
print(xrange.size())
print(yrange.size())
xcoord = xrange.repeat(yrange.size()[0])
ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
print(xcoord.size())
print(ycoord.size())
grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)
print(grid)