import torch
from torch import nn #nn  contains all of the PyTorch's building blocks for neural network
import matplotlib.pyplot as plt

# Check PyTorch version
# print(torch.__version__)

# what_were_covering = {1: 'data (prepare and load)',
#                       2: 'build model',
#                       3: 'fitting the model to data(training)',
#                       4: 'making predictions and evaluating model (inference)',
#                       5: 'saving and loading a model',
#                       6: 'putting it all together'
#                       }

# Data (Preparing and loading)
# ML is a game of two parts
# 1. Get data into a numerical representation
# 2. Build a model to learn patterns in that numerical representation

# We will us a linear regression formula to make a straight line with known parameters

# Create a known parameters
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim = 1)
y = weight * X + bias

# print(X[:10])
# print(y[:10])
# print(len(X))
# print(len(y))

# Spliting data into training and test sets
# Create training/test split

traing_split = int(0.8 * len(X))
X_train = X[:traing_split]
y_train = y[:traing_split]
X_test = X[traing_split:]
y_test = y[traing_split:]
# print(len(X_train), len(X_test))
# print(len(y_train), len(y_test))

def plot_prediction(train_data = X_train, train_labels = y_train, test_data = X_test, test_labels = y_test, predictions = None):
    plt.figure(figsize = (10, 7))
    
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c = "b", s = 4, label = "Training Data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s = 4, label = "Testing data")

    # Are there prediction?
    if predictions is not None:
        # Plot the predictions if they exist
        plt.scatter(test_data, predictions, c = "r", s = 4, label = "Predictions")

    # Show the legend
    plt.legend(prop = {"size": 14})
    # plt.show()
plot_prediction()

# Building model
# Create linear regression model class

# torch.nn -> contains all of the buildings for the neural network
# torch.nn.Parameter -> what parameters should our model try and learn, often a PyTorch layer from torch.nn will set these for us
# torch.nn.Module -> Base class for all NN modules, if you subclass it, you should overwrite forward()
# torch.optim -> this where the optimizers in PyTorch live, they will help with Gradient Descent
# def forward() ->  All nn.Module subclasses require you to overwrite forward(), this method defines what happens in the forward computation

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad = True, dtype = float))

    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
torch.manual_seed(42)

model_0 = LinearRegressionModel()

# Checkout the parameters
# print(list(model_0.parameters()))

# List the named parameters
print(model_0.state_dict())