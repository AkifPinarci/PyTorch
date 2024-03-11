import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
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

# Build a model
# 1. Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# 2. Construct a model
class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create two layers capeble of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # Takes in 2 features and upscale it to 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # Takes in 5 feature from prev layer and outputs a single layer(same shape as y)

# 3. Define a loss function and optimizer
    def forward(self, x):
        return self.layer_2(self.layer_1(x))
# 4. Create a training and test loop
model_0 = CircleModel().to(device)
# print(next(model_0.parameters()).device)

# Faster way of constructing simple model
# model_0 = nn.Sequential(
#     nn.Linear(in_features=2, out_features=5),
#     nn.Linear(in_features = 5, out_features = 1)
# ).to(device)
# print(model_0)

# print(model_0.state_dict())

# Make untrained prediciton
with torch.inference_mode():
    untrained_predictions = model_0(X_test.to(device))

print(f"Length of predictions: {len(untrained_predictions)}, Shape: {untrained_predictions.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_predictions[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")