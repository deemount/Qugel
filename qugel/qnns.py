import os
import re
from typing import Any, List, Mapping, Union

from pennylane import numpy as np
import pennylane as qml
import torch

from torch import nn
from torch.nn import functional as F

from pennylane.templates import RandomLayers

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import torch
from qgates import *


def print_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    # print(model)    
    print("The number of parameters: {}".format(num_params))


def Q_count_parameters(qnn):
    # print(dict(qnn.named_parameters()))
    for name, param in qnn.named_parameters():
        param.requires_grad = True
        print(name, param.data)
    return sum(p.numel() for p in qnn.parameters() if p.requires_grad)


def freeze_until(net, param_name):
    [k for k, v in net.named_parameters() if v.requires_grad]

    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


#extracting features from the images
class QCONV1(nn.Module):
    def __init__(self, n_qubits, n_layers, circuit, dev_train, gpu_device, patch_size, img_size_single, num_classes):
        super().__init__()
        self.n_qubits = n_qubits
        self.patch_size = patch_size
        self.H = img_size_single
        self.n_layers = n_layers
        self.device = gpu_device
        self.circuit = circuit
        self.num_classes = num_classes
        self.dev_train = dev_train
        self.classifier = torch.nn.Sequential(
            nn.Linear(4608, self.num_classes),
        )
        # self.q_params = nn.Parameter(torch.Tensor(self.n_qubits, self.n_qubits))
        self.q_params = nn.Parameter(0.001 * torch.randn(self.n_qubits))
        self.pqc = qml.QNode(circuit, self.dev_train, interface='torch')
        # Q_Plot(self.pqc, self.n_qubits, self.n_layers)


    def forward(self, X):
        assert len(X.shape) == 4, "Input X should have shape (batch_size, channels, height, width)"
        batch_size, channels, H, W = X.shape
        print(X.shape)
        patch_size = 2
        X_out = []

        for i in range(0, batch_size):
            for j in range(0, H, patch_size):
                for k in range(0, W, patch_size):
                    # Get 2x2 pixels and make them a 1D array
                    patch = X[i, :, j:j + patch_size, k:k + patch_size].flatten()
                    m = self.pqc(patch).float().unsqueeze(0)
                    X_out.append(m)

        X_out = torch.stack(X_out)  # Convert list to tensor
        print("X shape={}, type={}".format(X_out.shape, type(X_out)))
        X_out = X_out.view(X_out.size(0), -1)
        print ('X_out.data.shape:',X_out.data.shape)
        X_out = self.classifier(X_out.flatten())

        # X_out = self.classifier(X_out.view(-1, self.num_classes))
        print('X_out.data.shape:', X_out.data.shape)
        return X_out

# Define the Quanvolutional Neural Network
class QuanvolutionalNeuralNetwork(nn.Module):
    def __init__(self, n_qubits, n_layers, circuit, dev_train, gpu_device, patch_size, img_size_single, num_classes):
        super().__init__()
        self.n_qubits = n_qubits
        self.patch_size = patch_size
        self.img_size_single = img_size_single
        self.n_layers = n_layers
        self.device = gpu_device
        self.circuit = circuit
        self.num_classes = num_classes
        self.dev_train = dev_train

        print(
            f"n_qubits={self.n_qubits}, n_layers={self.n_layers}, circuit={self.circuit}, dev_train={self.dev_train}, gpu_device={self.device}, patch_size={self.patch_size}, img_size_single={self.img_size_single}, num_classes={self.num_classes}")

        self.fc1 = nn.Linear(self.n_qubits, self.num_classes)
        # self.q_params = nn.Parameter(torch.Tensor(self.n_qubits, self.n_qubits))
        self.q_params = nn.Parameter(0.001 * torch.randn(self.n_qubits,self.n_qubits))
        # for param in self.q_params:
        #     if param.requires_grad==True:
        #         print ('*****TRUE******')
        # self.lr1 = nn.LeakyReLU(0.3)
        # nn.init.xavier_uniform_(self.q_params)
        # qnode = qml.QNode(circuit, self.dev_train, interface = 'torch')
        self.pqc = qml.QNode(circuit, self.dev_train, interface='torch')
        weight_shapes = {"q_weights": (n_qubits, n_qubits),'n_qubits': self.n_qubits, 'q_depth': self.n_layers}
        # inputs, q_weights, n_qubits, q_depth
        # self.ql1 = qml.qnn.TorchLayer(self.pqc, weight_shapes)
        Q_Plot(self.pqc, self.n_qubits, self.n_layers)

    def extract_patches(self, x):
        patches = []
        bs, c, w, h = x.size()
        for i in range(w - self.patch_size + 1):
            for j in range(h - self.patch_size + 1):
                patch = x[:, :, i:i + self.patch_size, j:j + self.patch_size]
                patches.append(patch)
        patches = torch.stack(patches, dim=1).view(bs, -1, c * self.patch_size * self.patch_size)
        return patches

    def forward(self, x):
        # x = torch.tanh(x) * np.pi / 2.0
        # x=x/255
        assert len(x.shape) == 4  # (bs, c, w, h)
        bs = x.shape[0]  # batch_size = x.size(0)
        c = x.shape[1]  # RGB or mono 
        x = x.view(bs, c, self.img_size_single, self.img_size_single)
        q_in = self.extract_patches(x)
        q_in = q_in.to(self.device)
        print (q_in.shape) #torch.Size([32, 16129, 12])
        # q_out = torch.Tensor(0, n_qubits)
        q_out = torch.Tensor(0, self.n_qubits)

        # XL = []
        X = q_out.to(self.device)
        # print (q_in.shape) #torch.Size([8, 16129, 12])
        for elem in q_in:
            # print ('shapes:{},{},{},{}'.format(elem.shape, self.q_params.shape,self.n_qubits, self.n_layers))
            # output = torch.stack([torch.hstack(circuit(x, params)) for x in input])
            #PennyLane hard codes the names ... 'q_weights', 'n_qubits', and 'q_depth'
            q_out_elem = self.pqc(elem, q_weights=self.q_params, n_qubits=self.n_qubits, q_depth=self.n_layers).float().unsqueeze(0)
            # q_out_elem = self.pqc(elem).float().unsqueeze(0)
            print(q_out_elem.shape)
            X=torch.hstack(tuple(q_out_elem))
            # XL.append(q_out_elem)
            # X = torch.cat((X, q_out_elem))
        # X = torch.stack(XL, dim=0)
        # Reshape X to match the subsequent layer's input requirements
        # print (X.shape)
        print(type(X))  # <class 'torch.Tensor'>
        x = self.fc1(X.view(-1, self.n_qubits))
        # x = self.fc1(x)

        return x


class Quantumnet(nn.Module):
    def __init__(self, n_features, q_pqc, n_qubits, q_depth, q_delta, device, num_classes):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.num_classes = num_classes
        self.q_pqc = q_pqc
        self.q_net_block_name = self.q_pqc.__name__

        # self.pre_net = nn.Linear(512, n_qubits) # resnet 18
        # self.entpower= qml.entangling_power(q_net)
        self.pre_net = nn.Linear(n_features, n_qubits)  # seresnet 50

        self.q_params = nn.Parameter(q_delta * torch.randn(self.q_depth * self.n_qubits))
        self.post_net = nn.Linear(self.n_qubits, self.num_classes)

        self.device = device

    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, self.n_qubits)
        q_out = q_out.to(self.device)
        for elem in q_in:
            q_out_elem = self.q_pqc(elem, self.q_params, self.n_qubits, self.q_depth).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        return self.post_net(q_out)

# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print (device)

#     n_qubits_lst_tst = [4]  # Number of qubits
#     q_depth_lst_tst = [2]  # Depth of the quantum circuit (number of variational layers)

#     for qq in (n_qubits_lst_tst):
#         n_qubits = qq
#         for dd in (q_depth_lst_tst):
#             q_depth = dd
#             dev_test = qml.device('default.qubit', wires=n_qubits)
#             @qml.qnode(dev_test, interface='torch')
#             def quantum_net_test(q_input_features, q_weights, n_qubits, q_depth):
#                 #     q_weights = q_weights.reshape(n_qubits, q_depth)
#                 Q_encoding_block(q_input_features, n_qubits)
#                 Q_quanvol_block_C(q_weights, n_qubits, q_depth)

#                 exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
#                 return tuple(exp_vals)
#             Q_Plot(quantum_net_test, n_qubits, q_depth)
