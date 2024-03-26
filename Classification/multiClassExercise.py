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
import numpy as np

NUM_SAMPLES = 1000
RANDOM_SEED = 42
device = "cuda" if torch.cuda.is_available else "cpu"

np.random.seed(RANDOM_SEED)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
# plt.show()

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

class ExerciseModel(nn.Module):
    def __init__(self,inf, outf, hidden):
       super().__init__()
       self.layers = nn.Sequential(
          nn.Linear(in_features=inf, out_features=hidden),
          nn.ReLU(),
          nn.Linear(in_features=hidden, out_features=hidden),
          nn.ReLU(),
          nn.Linear(in_features=hidden, out_features=outf),
       )
    def forward(self, data):
       return self.layers.forward(data)
    
model_1 = ExerciseModel(2, 4, 2048).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model_1.parameters(), lr = 0.01)
acc_fn = Accuracy(task = "multiclass", num_classes = 4).to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 200

for epoch in range(epochs):
    model_1.train()  

    y_logits = model_1(X_train)
    y_probs = torch.softmax(y_logits, dim = 1)
    y_preds = torch.argmax(y_probs, dim = 1)

    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_train, y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test)
        test_pred = torch.softmax(test_logits, dim = 1).argmax(dim = 1)

        test_loss = loss_fn(test_logits, test_pred)
        test_accuracy = acc_fn(y_test, test_pred)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc} | Test loss: {test_loss}, Test accuracy: {test_accuracy}")





plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
plt.show()