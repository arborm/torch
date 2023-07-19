import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import math
import numpy as np

import random
#import torch.optim.lr_scheduler
from scheduler import WarmupCosineLR

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))
loss_fn = torch.nn.MSELoss(reduction = 'sum')

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = WarmupCosineLR(optimizer, warmup_epochs=3,
                               max_epochs=10)

for epoch in range(20):
    for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    scheduler.step()