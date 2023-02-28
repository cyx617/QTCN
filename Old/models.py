


import numpy as np


import torch
import torch.nn as nn
from modules import Qdense,Qconv1d,Qconv1d_2,Qconv1d_3,QSE







class QTCN(nn.Module):
    def __init__(self,device= "lightning.qubit"):
        super(QTCN, self).__init__()

        self.qconv1d_1 = Qconv1d(device=device, wires=2, circuit_layers=1, dilation=1, seed=None) # 量子标准卷积，卷积核大小为2
        self.qconv1d_2 = Qconv1d_2(device=device, wires=2, circuit_layers=1, dilation=1, seed=None) # 量子空洞卷积，膨胀因子为2，卷积核大小为2
        self.qconv1d_3 = Qconv1d_3(device=device, wires=1, circuit_layers=1, dilation=None, seed=None) # 量子标准卷积，卷积核大小为1
        self.qse = QSE(device=device, wires=2, circuit_layers=1, seed=None) # 量子SE模块
        self.Linear = nn.Linear(2, 1)  # 经典全连接


    def forward(self, x):

        x = torch.transpose(x,1,2) # (N,3,L)

        x1 = self.qconv1d_1(x) # (N,3,L-1)


        x1 = self.qconv1d_2(x1) # (N,2,L-3)

        # residual block
        resnet=True
        if resnet:
            x_skip = self.qconv1d_3(x)[:,:,-x1.shape[2]:] # (N,2,L-3)
            x_skip = self.qse(x_skip)  # (N,2,L-3)
            x1 = (x1 + x_skip)  # (N,2,L-3)

        x1 = x1[:,:,-1]  # (N,2)

        x1 = self.Linear(x1) # (N,1)

        return x1
