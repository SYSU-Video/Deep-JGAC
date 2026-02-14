import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.utils import PointNetFeaturePropagation
class HyperDecoder(nn.Module):
    def __init__(self):
        super(HyperDecoder, self).__init__()
        self.up1=PointNetFeaturePropagation(6,12) #in_channel,out_channel,npoint, radius, nsample,  radius1=100,nsample1=3
        self.up2=PointNetFeaturePropagation(12,12) #in_channel,out_channel,npoint, radius, nsample,  radius1=100,nsample1=3
        self.fc1=torch.nn.Linear(12,24)
        self.fc2=torch.nn.Linear(24,8)
        self.fc3=torch.nn.Linear(24,8)

    def forward(self,xyz0,xyz1,xyz2, x):
        x = self.up1(xyz1,xyz2,x)
        x = self.up2(xyz0,xyz1,x)
        x = torch.relu(self.fc1(x))
        x1 = torch.relu(self.fc2(x))
        x2 = torch.relu(self.fc3(x))
        # x = F.relu(self.fc3(x))
        return x1,x2
