import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.utils import Point_Conv

class HyperEncoder(nn.Module):
    def __init__(self, input_dim,  nsample,  nsample1):
        super(HyperEncoder, self).__init__()
        self.input_dim = input_dim
        self.sp1 = Point_Conv(input_dim, input_dim,  nsample,
                              nsample1)  # in_channel,out_channel,npoint, radius, nsample,  radius1=100,nsample1=3
        self.sp2 = Point_Conv(input_dim, input_dim,   nsample,
                              nsample1)  # in_channel,out_channel,npoint, radius, nsample,  radius1=100,nsample1=3
        self.sa1=torch.nn.Linear(input_dim,6)
        self.sa2=torch.nn.Linear(6,6)
        self.sa3=torch.nn.Linear(6,6)

    def forward(self, xyz,x):
        # x = F.relu(self.sp1(xyz,x))
        a=self.sp1(xyz,x)
        x=(a[1])
        aa=a[0]
        a = self.sp2(a[0],x)
        aaa=a[0]
        x = (a[1])
        x=F.relu(self.sa1(x))
        x=F.relu(self.sa2(x))
        x=F.relu(self.sa3(x))

        return aa,aaa,x

