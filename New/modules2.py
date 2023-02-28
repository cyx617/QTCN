import torch
from torch import nn

import torchvision

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import time
import os





def Q_H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """

    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def Q_RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)



def Q_entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])

# Convert (N,L) to (N,L-dilation)
class Qfilter(nn.Module):
    def __init__(self, device="default.qubit", wires=2, circuit_layers=1, dilation=1):
        super(Qfilter, self).__init__()
        self.wires = wires
        self.circuit_layers = circuit_layers
        self.dev = qml.device(device, wires=self.wires)
        self.dilation = dilation
        self.q_para = torch.nn.Parameter(torch.randn((self.circuit_layers, self.wires)))

    def forward(self, x):

        @qml.qnode(device=self.dev, interface="torch",diff_method='best')
        def circuit(inputs, weights):
            Q_H_layer(self.wires)
            Q_RY_layer(inputs)
            for k in range(self.circuit_layers):
                Q_entangling_layer(self.wires)
                Q_RY_layer(weights[k])
            return [qml.expval(qml.PauliZ(j)) for j in range(self.wires)]

        bs, t= x.size()
        kernel_size = self.wires
        if kernel_size > 1:
            t_out = t - self.dilation
            batch_input = torch.zeros((bs, t_out,kernel_size)) #(bs,t_out,k)
            for b in range(bs):
                for i in range(t_out):
                    batch_input[b,i,:] = torch.Tensor([x[b,i],x[b,i+self.dilation]])
            batch_input = torch.flatten(batch_input,0,1) # (bs*t_out,k)
            batch_input = torch.transpose(batch_input,0,1) # (k,bs*t_out)
            # quantum circuit executions with quantum parameter broadcasting
            out = circuit(batch_input,self.q_para).float() # (bs*t_out,k)
            out = torch.unflatten(out,0,(bs,t_out)) # (bs,t_out,k)
            out = torch.sum(out,dim=2) #(bs,t_out)
        else:
            # kernel_size = 1
            t_out = t
            batch_input = torch.flatten(x,0,1) #(bs*t_out,)
            batch_input = torch.reshape(batch_input,(1,-1)) #(1,bs*t_out)
            out = circuit(batch_input,self.q_para).float() #(bs*t_out,)
            out = torch.unflatten(out,0,(bs,t_out)) #(bs,t_out)

        return torch.reshape(out,(out.shape[0],1,out.shape[1]))  # (bs,1,t_out)





# 1-d quantum dilated convolution (in_channels=3,out_channels=1,dilation)
class Qfilter_multichannel(nn.Module):
    def __init__(self,device="default.qubit", wires=2, circuit_layers=1, dilation=1):
        super().__init__()

        self.qfilter1 = Qfilter(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation)
        self.qfilter2 = Qfilter(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation)
        self.qfilter3 = Qfilter(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation)

    def forward(self, x):

        out1 = self.qfilter1(x[:,0,:]) # (N,1,L-dilation)
        out2 = self.qfilter2(x[:,1,:]) # (N,1,L-dilation)
        out3 = self.qfilter3(x[:,2,:]) # (N,1,L-dilation)

        out = out1+out2 +out3  # (N,1,L-dilation)


        return out


# 1-d quantum dilated convolution (in_channels=3,out_channels=3,kernel_size=2, dilation=dilation)
class Qconv1d(nn.Module):
    def __init__(self,device="default.qubit", wires=2, circuit_layers=1, dilation=1):
        super().__init__()

        self.qconv1 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation)
        self.qconv2 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation)
        self.qconv3 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation)

    def forward(self, x):
        # x (N,3,L)
        out1 = self.qconv1(x) # (N,1,L-dilation)
        out2 = self.qconv2(x) # (N,1,L-dilation)
        out3 = self.qconv3(x) # (N,1,L-dilation)

        out = torch.cat((out1,out2,out3),1)  # (N,3,L-dilation)

        return out



# 1-d quantum dilated convolution (in_channels=3,out_channels=2,kernel_size=2, dilation=dilation)
class Qconv1d_2(nn.Module):
    def __init__(self,device="default.qubit", wires=2, circuit_layers=1, dilation=1):
        super().__init__()

        self.qconv1 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation)
        self.qconv2 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation)
        #self.qconv3 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)

    def forward(self, x):
        # x (N,3,L)
        out1 = self.qconv1(x) # (N,1,L-dilation)
        out2 = self.qconv2(x) # (N,1,L-dilation)
        out = torch.cat((out1,out2),1)  # (N,3,L-dilation)

        return out



# 1-d 1 by 1 quantum convolution (in_channels=3,out_channels=2,kernel_size=1)
class Qconv1d_3(nn.Module):
    def __init__(self,device="default.qubit", wires=1, circuit_layers=1, dilation=None):
        super().__init__()

        self.qconv1 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation)
        self.qconv2 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation)
        #self.qconv3 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)

    def forward(self, x):
        # x (N,3,L)
        out1 = self.qconv1(x) # (N,1,L-dilation)
        out2 = self.qconv2(x) # (N,1,L-dilation)
        out = torch.cat((out1,out2),1)  # (N,3,L-dilation)

        return out

# Quantum dense layer in QSE module
class Qdense(nn.Module):
    def __init__(self, device="default.qubit", wires=2, circuit_layers=1):
        super(Qdense, self).__init__()

        self.wires = wires
        self.circuit_layers = circuit_layers
        self.dev = qml.device(device, wires=self.wires)
        self.q_para = torch.nn.Parameter(torch.randn((self.circuit_layers, self.wires)))

    def forward(self, x):

        @qml.qnode(device=self.dev, interface="torch",diff_method='best')
        def circuit(inputs, weights):

            Q_H_layer(self.wires)
            Q_RY_layer(inputs)

            for k in range(self.circuit_layers):
                Q_entangling_layer(self.wires)
                Q_RY_layer(weights[k])

            return [qml.expval(qml.PauliZ(j)) for j in range(self.wires)]

        batch_input = torch.transpose(x,0,1) # (features,bs)
        out = circuit(batch_input,self.q_para).float() # (bs,features)

        return out



# QSE module
class QSE(nn.Module):
    def __init__(self,device="default.qubit", wires=2, circuit_layers=1):
        super().__init__()
        self.qdense = Qdense(device=device, wires=wires, circuit_layers=circuit_layers) # qdense

    def forward(self, x):
        # x (N,2,L)
        x_se = torch.mean(x,dim=2) # (N,2)
        #print(x_se.shape)
        x_se = self.qdense(x_se) # (N,2)
        #print(x_se.shape)
        if len(x_se.shape)==1:
            x_se = torch.reshape(x_se,(1,x_se.shape[0],1))
        else:
            x_se = torch.reshape(x_se,(x_se.shape[0],x_se.shape[1],1)) # (N,2,1)
        x = torch.mul(x,x_se) # (N,2,L)

        return x
