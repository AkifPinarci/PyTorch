import torch
from torch import nn #nn  contains all of the PyTorch's building blocks for neural network
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
MODEL_PATH = Path("Model")
MODEL_PATH.mkdir(parents = True, exist_ok = True)
MODEL_NAME = "O1_pt_model_exercise.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
WEIGHT = 0.3
BIAS = 0.9

class LinearRegressionModelExercise(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(data)
    
def create_data(start, end, step, device):
    X = torch.arange(start, end, step, device = device).unsqueeze(dim = 1)
    y = X * WEIGHT + BIAS
    train_split = int(0.8 * len(X))
    X_train = X[:train_split]
    y_train = y[:train_split]
    X_test = X[train_split:]
    y_test = y[train_split:]
    return X_train, y_train, X_test, y_test

def plot_prediction(train_data , train_labels, test_data, test_labels, predictions = None):
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

def model_saver(model):
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

def model_loader():
    try:
        torch.manual_seed(42)
        # Load the saved model for evaluation
        model = LinearRegressionModelExercise()
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    except:
        torch.manual_seed(42)
        model = LinearRegressionModelExercise()
    model.eval()
    return model

def train_model(loss_fn, optimizer, model, X_train, y_train, X_test, y_test, epochs):
    torch.manual_seed(42)

    for epoch in range(epochs):
        model.train()

        # 1. Forward Pass
        y_pred = model(X_train)

        # 2. Calculate the loss
        loss = loss_fn(y_pred, y_train)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Perform Backpropagation
        loss.backward()

        # 5. Optimizer Step
        optimizer.step()
    
        # Testing
        model.eval()
        with torch.inference_mode():
            test_prep = model(X_test)

            test_loss = loss_fn(test_prep, y_test)

        # if epoch % 10 == 0:
        #     print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train, y_train, X_test, y_test = create_data(0, 1, 0.01, device)
    model_1 = model_loader()
    model_1.to(device)
    print(model_1)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params = model_1.parameters(), lr = 0.001)
    train_model(loss_fn, optimizer, model_1, X_train, y_train, X_test, y_test, 1)

    model_1.eval()
    with torch.inference_mode():
        y_preds = model_1(X_test)
    plot_prediction(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), y_preds.cpu())
    print(model_1.state_dict())

    model_saver(model_1)