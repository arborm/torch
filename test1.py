import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

train_data = datasets.FashionMNIST(root = "data", train = True, download = True, transform = ToTensor())
test_data = datasets.FashionMNIST(root = "data", train = False, download = True, transform = ToTensor())

#batch_size = 64

#train_dataloader = DataLoader(train_data, batch_size = batch_size)
#test_dataloader = DataLoader(test_data, batch_size = batch_size)

labels_map = {0: "a", 1: "a", 2: "a", 3:"a", 4:"a", 5:"a", 6:"a", 7:"a", 8:"a", 9:"a"}
figure = plt.figure(figsize = (10,10))

for i in range(9):
    sample_index = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_index]
    figure.add_subplot(3, 3, i+1)
    plt.title(labels_map[label])
    plt.imshow(img.squeeze())
plt.show()



figure.savefig('temp.png')