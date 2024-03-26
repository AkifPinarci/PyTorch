# To import some functions from another module
import sys
import os
path = os.path.abspath("Helpers")
sys.path.append(path)
from helper_functions import plot_predictions, plot_decision_boundary


import sklearn
from sklearn.datasets import make_circles, make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import requests
from pathlib import Path 
from torchmetrics import Accuracy
from sklearn.datasets import make_moons

NUM_SAMPLES = 1000
RANDOM_SEED = 42
device = "cuda" if torch.cuda.is_available else "cpu"

X, y = make_moons(n_samples=NUM_SAMPLES,
                  noise=0.07,
                  random_state=RANDOM_SEED)

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

class ExerciseModel(nn.Module):
    def __init__(self, inf, outf, hidden):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=inf, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=outf)
        )
    
    def forward(self, data):
        return self.linear_layer.forward(data)


model_0 = ExerciseModel(2, 1, 32).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr = 0.1)
acc_fn = Accuracy(task="multiclass", num_classes=2).to(device)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epochs = 1000

for epoch in range(epochs):
    model_0.train()

    y_logits = model_0(X_train).squeeze()
    y_probs = torch.sigmoid(y_logits)
    y_preds = torch.round(y_probs)

    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_preds, y_train)
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        y_test_logits = model_0(X_test).squeeze()
        y_test_probs = torch.sigmoid(y_test_logits)
        y_test_preds = torch.round(y_test_probs)

        test_loss = loss_fn(y_test_logits, y_test)
        test_acc = acc_fn(y_test_preds, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.2f} Acc: {acc:.2f} | Test loss: {test_loss:.2f} Test acc: {test_acc:.2f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()