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
    plt.show()

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
    




# Checkout the parameters
# print(list(model_0.parameters()))

# List the named parameters
# print(model_0.state_dict())

# Making predictions using 'torch.inference_mode()'
# When we pass data through our model, it's going to run it through the forward() method.

# Make predictions with model
# with torch.inference_mode():
#     y_preds = model_0(X_test)

# print(y_preds)

# The whole idea of training is for a model to move from some unknown parameters to some known parameters.
# Loss function may also be called cost function
# Loss function: How wrong is our model's predictions are to the idela outputs, (lower is better)
# Optimizer: Takes into account the loss of a model and adjusts the model's parameters(eg. weights & bias in our case) to improve the loss function



# Build a training loop (and a testing loop) in PyTorch
# Couple of things we need in a training loop
# 0. Loop through the data 
# 1. Forward pass - also called forward propogation
# 2. Calculate the loss 
# 3. Optimizer zero grad
# 4. Loss backwards
# 5. Optimizer step

# An epoch is one loop through the data - (this is a hyperparameter because we've set it)
def train(epochs, loss_fn, optimizerm, model_0):

    epoch_count = []
    loss_values = []
    test_loss_values = []

    # Training
    # 0. Loop through the data
    for epoch in range(epochs):
        # Set the model to training mode
        model_0.train() # train mode in PyTorch sets all parameters that require gradients to require gradients
        
        # 1. Forward pass
        y_preds = model_0(X_train)

        # 2. Calculate the loss
        loss = loss_fn(y_preds, y_train)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Perform backpropogation on the loss with respest to the parameters of the model
        loss.backward()

        # 5. Step the optimizer
        optimizer.step()

        # Testing
        model_0.eval()  # turns off gradient tracking(also turns of diffferend setting in the model not needed for evaluation/testing)

        with torch.inference_mode(): # turns of gradient tracking & a couple more things behind the scenes
            # 1. Do the forward pass
            test_pred = model_0(X_test)

            # 2. Calculate the loss
            test_loss = loss_fn(test_pred, y_test)

            # Print out what is happening
            if epoch % 10 == 0:
                epoch_count.append(epoch)
                loss_values.append(loss)
                test_loss_values.append(test_loss)
                print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
                print(model_0.state_dict())

def model_loader():
    try:
        # Load the saved model for evaluation
        model = LinearRegressionModel()
        model.load_state_dict(torch.load('final_model.pt'))
    except:
        model = LinearRegressionModel()
    return model

if __name__ == '__main__':
    torch.manual_seed(42)
    model = model_loader()
    # Setup a loss function
    loss_fn = nn.L1Loss()

    # Setup a optimizer
    optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01, momentum = 0.9)

    train(100, loss_fn, optimizer, model)

    with torch.inference_mode():
        y_preds_new = model(X_test)
    plot_prediction(predictions=y_preds_new)
    torch.save(model.state_dict(), 'final_model.pt')

    

