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

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# Create multi class data
X_blob, y_blob = make_blobs(n_samples = 1000, 
                              n_features= NUM_FEATURES,
                              centers = NUM_CLASSES,
                              cluster_std = 1,
                              random_state = RANDOM_SEED
                              )

# Turn Data into Tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state = RANDOM_SEED)

# Plot data
# plt.figure(figsize = (10, 7))
# plt.scatter(X_blob[:, 0], X_blob[:, 1], c = y_blob, cmap = plt.cm.RdYlBu)
# plt.show()

device = "cuda" if torch.cuda.is_available else "cpu"
# print(y_blob_test.unsqueeze(dim = 1).shape)

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hiddent_units = 8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_features, hiddent_units),
            nn.Linear(hiddent_units, hiddent_units),
            nn.Linear(hiddent_units, output_features)
        )
    
    def forward(self, data):
        return self.linear_layer_stack.forward(data)

# print(torch.unique(y_blob_train))
    
model_4 = BlobModel(2, 4, 8).to(device)

# Create a loss function and optimizer
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params = model_4.parameters(), lr = 0.01)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = X_blob_train.to(device), X_blob_test.to(device), y_blob_train.to(device), y_blob_test.to(device)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# model_4.eval()
# with torch.inference_mode():
#     y_logits = model_4(X_blob_train)
# # Convert model's logit outputs to predcition probabilities
# y_pred_probs = torch.softmax(y_logits, dim = 1)
# print(y_logits[:5])
# print(y_pred_probs[:5])
# y_preds = torch.argmax(y_pred_probs, dim = 1)
# print(y_preds)
    

# Create a training loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100

for epoch in range(epochs):
    model_4.train()

    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim = 1).argmax(dim = 1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_blob_train, y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim = 1).argmax(dim = 1)

        test_loss = loss_fn(test_logits, test_pred)
        test_accuracy = accuracy_fn(y_blob_test, test_pred)
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc} | Test loss: {test_loss}, Test accuracy: {test_accuracy}")


model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)
    y_pred = torch.softmax(y_logits, dim = 1).argmax(dim = 1)

# plt.figure(figsize = (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_4, X_blob_train, y_blob_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_4, X_blob_test, y_blob_test)
# plt.show()
        
tmAcc = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device)
res = tmAcc(y_pred, y_blob_test)
print(res)