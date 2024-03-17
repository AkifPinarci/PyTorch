# To import some functions from another module
import sys
import os
path = os.path.abspath("Helpers")
sys.path.append(path)
from helper_functions import plot_predictions, plot_decision_boundary


import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import requests
from pathlib import Path 


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create circle data
n_samples = 1000
X, y = make_circles(n_samples, 
                    noise = 0.03, 
                    random_state= 42)

# Convert data to tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)


# Train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
class CircleModelV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=1)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, data):
        return (self.layer2(self.relu(self.layer1(data))))

# Model, loss, optimmizer creation
model3 = CircleModelV3()
model3 = model3.to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model3.parameters(), lr = 0.1)

# Accuracy function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Put data into target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


epochs = 100

# Training loop
for epoch in range(epochs):
    model3.train()
    y_logits = model3(X_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits)) #logits -> prediction probabilities -> prediction labels

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_preds)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

### Testing
    model3.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model3(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
      # 2. Calcuate loss and accuracy
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_true=y_test,
                             y_pred=test_pred)

   # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.5f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.5f}%")


model3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model3(X_test))).squeeze()



def getRes():
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model3, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model3, X_test, y_test)
    plt.show()
getRes()
