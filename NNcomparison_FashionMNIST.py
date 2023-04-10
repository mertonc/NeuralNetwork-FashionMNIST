# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 22:49:23 2022

@author: merto
"""

import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
    )

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            #nn.Dropout(0.3), nn.BatchNorm1d(512)
            nn.ReLU(), #also test with nn.Sigmoid() and without nonlinear activation functions
            nn.Linear(512, 512),
            #nn.Dropout(0.3), nn.BatchNorm1d(512)
            nn.ReLU(),
            nn.Linear(512, 10),
            )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

class SigmoidNeuralNetwork(nn.Module):
    def __init__(self):
        super(SigmoidNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10),
            )
        
    def forward(self, x):
        x = self.flatten(x)
        logits_2 = self.linear_sigmoid_stack(x)
        return logits_2
    
sigmodel = SigmoidNeuralNetwork().to(device)
print(sigmodel)

    
learning_rate = 1e-3
batch_size = 64
epochs = 5

#loss function
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer): #remove optimizer for baseline accuracy
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')
            
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')
    

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer_2 = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
optimizer_3 = torch.optim.Rprop(model.parameters(), lr=learning_rate)

start = time.time()
epochs = 10
for t in range(epochs):
    print(f'Epoch {t+1}\n--------------------')
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

end = time.time()
print("The time of execution of above program is :", (end-start) * 10**3, "ms")
# for c in range(epochs):
#     print(f'Epoch {t+1}\n--------------------')
#     train_loop(train_dataloader, sigmodel, loss_fn, optimizer)
#     test_loop(test_dataloader, sigmodel, loss_fn)

# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f'Predicted class: {y_pred}')

# input_image = torch.rand(3,28,28)
# print(input_image.size())

# flatten = nn.Flatten()
# flat_image = flatten(input_image)
# print(flat_image.size())

# layer1 = nn.Linear(in_features=28*28, out_features=20)
# hidden1 = layer1(flat_image)
# print(hidden1.size())

# for t in range(epochs):
#     print(f'Epoch {t+1}\n--------------------')
#     train_loop(train_dataloader, sigmodel, loss_fn, optimizer)
#     test_loop(test_dataloader, sigmodel, loss_fn)
    

# class TanhNeuralNetwork(nn.Module):
#     def __init__(self):
#         super(TanhNeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.tanh_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.Tanh(),
#             nn.Linear(512, 512),
#             nn.Tanh(),
#             nn.Linear(512, 10),
#             )
        
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.tanh_relu_stack(x)
#         return logits

# tanhmodel = TanhNeuralNetwork()


