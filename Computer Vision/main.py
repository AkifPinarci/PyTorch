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

