import torch
from torch import nn

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

train_data = datasets.FashionMNIST(
    root = "data", 
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = None
)

test_data = datasets.FashionMNIST(
    root = "data", 
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = None
)

image, label = train_data[0]
# print(image, label)
# print(train_data.classes)
# print(train_data.class_to_idx)   
# print(train_data.targets)
# print(image.shape)
# print(train_data.classes[label])

# plt.imshow(image.squeeze(), cmap = "gray")
# plt.title(train_data.classes[label])
# plt.axis(False)
# plt.show()

torch.manual_seed(1)
fig = plt.figure(figsize = (9, 9))
rows, cols, = 4, 4
for i in range(1, rows * cols + 1):
    random_index = torch.randint(0, len(train_data), size = [1]).item()
    img, label = train_data[random_index]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap = "gray")
    plt.title(train_data.classes[label])
    plt.axis(False)
print(train_data)