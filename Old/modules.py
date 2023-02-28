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
    def __init__(self, device="default.qubit", wires=2, circuit_layers=1, dilation=1, seed=None):
        super(Qfilter, self).__init__()


        self.wires = wires
        self.dev = qml.device(device, wires=self.wires)

        #self.stride = stride
        self.dilation = dilation


        if seed is None:
            seed = np.random.randint(low=0, high=10e6)

        print("Initializing Circuit with random seed", seed)


        @qml.qnode(device=self.dev, interface="torch",diff_method='best')
        def circuit(inputs, weights):
            n_inputs=2

            Q_H_layer(self.wires)
            Q_RY_layer(inputs)

            for k in range(circuit_layers):
                Q_entangling_layer(self.wires)
                Q_RY_layer(weights[k])

            return [qml.expval(qml.PauliZ(j)) for j in range(self.wires)]

        weight_shapes = {"weights": [circuit_layers, wires]}
        self.circuit = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)


    def forward(self, x):
        bs, t= x.size()


        kernel_size = self.wires
        if kernel_size > 1:
            t_out = t - self.dilation



            out = torch.zeros((bs, t_out))   # N,L


            for b in range(bs):
                for i in range(t_out):
                    input = torch.Tensor([x[b,i],x[b,i+self.dilation]])
                    q_results = torch.sum(self.circuit(input))
                    #print('q_results:',b,q_results)
                    out[b,i] = q_results
        else:
            t_out = t
            out = torch.zeros((bs, t_out))
            for b in range(bs):
                for i in range(t_out):
                    input = torch.Tensor([x[b,i]])
                    q_results = self.circuit(input)
                    out[b,i] = q_results



        return torch.reshape(out,(out.shape[0],1,out.shape[1]))  # (N,1,L)





# 1-d quantum dilated convolution (in_channels=3,out_channels=1,dilation)
class Qfilter_multichannel(nn.Module):
    def __init__(self,device="default.qubit", wires=2, circuit_layers=1, dilation=1, seed=None):
        super().__init__()

        self.qfilter1 = Qfilter(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)
        self.qfilter2 = Qfilter(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)
        self.qfilter3 = Qfilter(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)

    def forward(self, x):
        # 输入x的尺寸为(N,3,L)

        out1 = self.qfilter1(x[:,0,:]) # (N,1,L-dilation)
        out2 = self.qfilter2(x[:,1,:]) # (N,1,L-dilation)
        out3 = self.qfilter3(x[:,2,:]) # (N,1,L-dilation)

        out = out1+out2 +out3  # (N,1,L-dilation)


        return out


# 1-d quantum dilated convolution (in_channels=3,out_channels=3,kernel_size=2, dilation=dilation)
class Qconv1d(nn.Module):
    def __init__(self,device="default.qubit", wires=2, circuit_layers=1, dilation=1, seed=None):
        super().__init__()

        self.qconv1 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)
        self.qconv2 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)
        self.qconv3 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)

    def forward(self, x):
        # x (N,3,L)
        out1 = self.qconv1(x) # (N,1,L-dilation)
        out2 = self.qconv2(x) # (N,1,L-dilation)
        out3 = self.qconv3(x) # (N,1,L-dilation)

        out = torch.cat((out1,out2,out3),1)  # (N,3,L-dilation)

        return out



# 1-d quantum dilated convolution (in_channels=3,out_channels=2,kernel_size=2, dilation=dilation)
class Qconv1d_2(nn.Module):
    def __init__(self,device="default.qubit", wires=2, circuit_layers=1, dilation=1, seed=None):
        super().__init__()

        self.qconv1 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)
        self.qconv2 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)
        #self.qconv3 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)

    def forward(self, x):
        # x (N,3,L)
        out1 = self.qconv1(x) # (N,1,L-dilation)
        out2 = self.qconv2(x) # (N,1,L-dilation)
        #out3 = self.qconv3(x) # (N,1,L-dilation)

        out = torch.cat((out1,out2),1)  # (N,3,L-dilation)

        return out



# 1-d 1 by 1 quantum convolution (in_channels=3,out_channels=1,kernel_size=1)
class Qconv1d_3(nn.Module):
    def __init__(self,device="default.qubit", wires=1, circuit_layers=1, dilation=None, seed=None):
        super().__init__()

        self.qconv1 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)
        self.qconv2 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)
        #self.qconv3 = Qfilter_multichannel(device=device, wires=wires, circuit_layers=circuit_layers, dilation=dilation, seed=seed)

    def forward(self, x):
        # x (N,3,L)
        out1 = self.qconv1(x) # (N,1,L-dilation)
        out2 = self.qconv2(x) # (N,1,L-dilation)
        #out3 = self.qconv3(x) # (N,1,L-dilation)

        out = torch.cat((out1,out2),1)  # (N,3,L-dilation)

        return out

# Quantum dense layer in QSE module
class Qdense(nn.Module):
    def __init__(self, device="default.qubit", wires=2, circuit_layers=1, seed=None):
        super(Qdense, self).__init__()

        # init device
        self.wires = wires
        self.dev = qml.device(device, wires=self.wires)


        if seed is None:
            seed = np.random.randint(low=0, high=10e6)

        print("Initializing Circuit with random seed", seed)


        @qml.qnode(device=self.dev, interface="torch",diff_method='best')
        def circuit(inputs, weights):
            n_inputs=2

            Q_H_layer(self.wires)
            Q_RY_layer(inputs)

            for k in range(circuit_layers):
                Q_entangling_layer(self.wires)
                Q_RY_layer(weights[k])

            return [qml.expval(qml.PauliZ(j)) for j in range(self.wires)]

        weight_shapes = {"weights": [circuit_layers, wires]}
        self.circuit = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)


    def forward(self, x):
        bs, features= x.size()
        out = torch.zeros((bs, features))   # N,L

        # Loop over the coordinates of the top-left pixel of 2X2 squares
        for b in range(bs):
                input = torch.Tensor(x[b,:])
                q_results = self.circuit(input)
                out[b,:] = q_results

        return out # (N,1,L)



# QSE module
class QSE(nn.Module):
    def __init__(self,device="default.qubit", wires=2, circuit_layers=1, seed=None):
        super().__init__()
        #self.q_depth = q_depth
        self.qdense = Qdense(device=device, wires=wires, circuit_layers=circuit_layers, seed=seed) # 量子全连接

    def forward(self, x):
        # x (N,2,L)
        x_se = torch.mean(x,dim=2) # (N,2)
        x_se = self.qdense(x_se) # (N,2)
        x_se = torch.reshape(x_se,(x_se.shape[0],x_se.shape[1],1)) # (N,2,1)

        x = torch.mul(x,x_se) # (N,2,L)

        return x
