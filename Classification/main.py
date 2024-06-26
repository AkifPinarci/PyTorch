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

# Make classification data and get it ready
# Make 1000 samples

n_samples = 1000

X, y = make_circles(n_samples, 
                    noise = 0.03, 
                    random_state= 42)

# print(X[:5])
# print(y[:5])

# Make DataFrame of circle data
circles = pd.DataFrame({"X1" : X[:,0],
                        "X2": X[:,1],
                        "label" : y})

# print(circles.head(10))

# Show the circles
# plt.scatter(x = X[:, 0], 
#             y = X[:, 1],
#             c = y, 
#             cmap = plt.cm.RdYlBu)
# plt.show()


# print(X.shape, y.shape)
# print(type(X), type(y))

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
# print(X, y)
# Create train and test splits

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

split_index = int(0.8 * len(X))
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]
# print(len(y_train))
# print(len(y_test))

# # Build a model
# # 1. Setup device agnostic code
device = "cpu" if torch.cuda.is_available() else "cpu"
# # 2. Construct a model
# class CircleModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Create two layers capeble of handling the shapes of our data
#         self.layer_1 = nn.Linear(in_features=2, out_features=5) # Takes in 2 features and upscale it to 5 features
#         nn.ReLU()
#         self.layer_2 = nn.Linear(in_features=5, out_features=1) # Takes in 5 feature from prev layer and outputs a single layer(same shape as y)
# # 3. Define a loss function and optimizer
#     def forward(self, data):
#         return self.layer_2(self.layer_1(data))
# # 4. Create a training and test loop
# model_0 = CircleModel().to(device)
# # print(next(model_0.parameters()).device)

# # Faster way of constructing simple model
# # model_0 = nn.Sequential(
# #     nn.Linear(in_features=2, out_features=5),
# #     nn.Linear(in_features = 5, out_features = 1)
# # ).to(device)
# # print(model_0)

# # print(model_0.state_dict())

# # Make untrained prediciton
# with torch.inference_mode():
#     untrained_predictions = model_0(X_test.to(device))

# # print(f"Length of predictions: {len(untrained_predictions)}, Shape: {untrained_predictions.shape}")
# # print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
# # print(f"\nFirst 10 predictions:\n{untrained_predictions[:10]}")
# # print(f"\nFirst 10 test labels:\n{y_test[:10]}")
    
# # Setup loss function and optimizer, !!! this is problem specific
# # For example, for regression MAE or MSE(mean absolute error or mean squared error)
# # For classification binary cross entropy or categorical cross entropy(cross entropy)
# loss_fn = nn.BCEWithLogitsLoss()

# optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.01)

# Calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# # Our model outputs are going to be raw logits
# # We can convert these logits into prediction probabilities by passing them to some kind of activation function (e.g. sigmoid for binary classification and softmax for multiclass classification)
# # Then we can convert our model's prediction probabilities ro prediction labels by either rounding them or taking the argmax()
# model_0.eval()
# with torch.inference_mode():
#     # logits -> pred probs -> pred labels
#     y_pred_label = torch.round(torch.sigmoid(model_0(X_test.to(device))))

# # # Use sigmoid activation function on our model logits
# # print(y_test[:5])

# X_train.to(device)
# X_test.to(device)
# y_train.to(device)
# y_test.to(device)
# model_0.to(device)
# # Train the model
# epochs = 0
# for epoch in range(epochs):
#     model_0.train()
#     # 1. Forward pass
#     y_logits = model_0(X_train).squeeze()   
#     y_pred = torch.round(torch.sigmoid(y_logits)) 
#     # print(y_logits)
#     # 2. Calculate the loss
#     loss = loss_fn(y_logits, y_train)
#     acc = accuracy_fn(y_train, y_pred)

#     # 3. Optimizer zero grad
#     optimizer.zero_grad()
#     # 4. Backward (backpropagation)
#     loss.backward()
#     # 5. Optimizer step
#     optimizer.step()

#     # Testing
#     model_0.eval()
#     with torch.inference_mode():
#         test_logits = model_0(X_test).squeeze()
#         test_pred = torch.round(torch.sigmoid(test_logits))

#         # Calculate loss and acc
#         test_loss = loss_fn(test_logits, y_test)
#         test_acc = accuracy_fn(y_test, test_pred)

#     if epoch % 10 == 0:
#         print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

def getRes():
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model_1, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model_1, X_test, y_test)
    plt.show()
# getRes()
    
# How to improve model
# 1. Add more layers - give the model more chances to learn about patterns in the DeprecationWarning
# 2. Add more hidden units - go from 5 hidden units to 10 hidden units
# 3. Fit for longer
# 4. Cahnging the activation function
# 5. Change the learning rate
# 6. Change the loss function

 
# Improve model
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, data):
        return self.layer_3(self.layer_2(self.layer_1(data)))
    
model_1 = CircleModelV1().to(device)
loss_fn1 = nn.BCEWithLogitsLoss()
optimizer1 = torch.optim.SGD(params = model_1.parameters(), lr = 0.1)
# print(model_1.state_dict())

torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 0
X_train.to(device)
y_train.to(device)
X_test.to(device)
y_test.to(device)

for epoch in range(epochs):
    model_1.train()


    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn1(y_logits, y_train)
    acc = accuracy_fn(y_true = y_train, y_pred=y_pred)

    optimizer1.zero_grad()

    loss.backward()

    optimizer1.step()

    # Testing
    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()

        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn1(test_logits, test_pred)
        test_acc = accuracy_fn(y_test, test_pred)

    if epoch % 100 == 0:
       print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
    
# getRes()
       
# Create some data
       
weigth = 0.7
bias = 0.3

start = 0
end = 1
step = 0.01

X_regression = torch.arange(start, end, step).unsqueeze(dim = 1)
y_regression = X_regression * weigth + bias
train_split = int(0.8 * len(X_regression))
X_train_regression = X_regression[:train_split]
y_train_regression = y_regression[:train_split]
X_test_regression = X_regression[train_split:]
y_test_regression = y_regression[train_split:]

model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features= 10),
    nn.Linear(in_features=10, out_features= 10),
    nn.Linear(in_features=10, out_features= 1),
)

loss_fn2 = nn.L1Loss()
optimizer2 = torch.optim.SGD(model_2.parameters(), lr= 0.01)


torch.manual_seed(42)

epochs = 1000

for epoch in range(epochs):
    model_2.train()

    y_pred = model_2(X_train_regression)
    loss = loss_fn2(y_pred, y_train_regression)

    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn2(test_pred, y_test_regression)

    if epoch % 100 == 0:
       print(f"Epoch: {epoch} | Loss: {loss:.5f}| Test loss: {test_loss:.5f}")

model_2.eval()
with torch.inference_mode():
    y_pred = model_2(X_test_regression)


plot_predictions(X_train_regression, y_train_regression, X_test_regression, y_test_regression, y_pred)
plt.show()