import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.cuda as cuda
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import sys

working_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(working_dir)

# Set random seed
torch.manual_seed(30)
np.random.seed(30)

# Device configuration
device = torch.device("cuda" if cuda.is_available() else "cpu")


# Generate 17 similar functions
def generate_simple_functions(num_functions, input_dim=5, output_dim=5):
    functions = []
    for _ in range(num_functions):
        # Generate random coefficients for each function
        coeffs = np.random.randn(input_dim, output_dim) * 0.1
        bias = np.random.randn(output_dim) * 0.1

        def func(x, coeff=coeffs, b=bias):
            return (
                np.dot(x, coeff) + b  # + np.sin(x).sum() * 0.1
            )  # Adding a non-linear term

        functions.append(func)
    return functions


def generate_polynomial_functions(
    num_functions, input_dim=5, output_dim=5, max_degree=3
):
    polyfunctions = []
    for _ in range(num_functions):
        # Generate random coefficients for each function and for each degree
        # coeffs shape will be (output_dim, input_dim, max_degree)
        # Each output dimension will have a different polynomial equation
        coeffs = np.random.randn(output_dim, input_dim, max_degree) * 0.1
        bias = np.random.randn(output_dim) * 0.1

        def poly_func(x, coeff=coeffs, b=bias):
            # x should be (n_samples, input_dim)
            # We calculate the polynomial for each input dimension and sum them
            y = np.zeros((x.shape[0], output_dim))
            for i in range(output_dim):
                for j in range(input_dim):
                    for d in range(max_degree):
                        # Compute x^d for the current degree, d+1 because range starts at 0
                        y[:, i] += coeff[i, j, d] * np.power(x[:, j], d + 1)
                y[:, i] += b[i]
            return y

        polyfunctions.append(poly_func)
    return polyfunctions


######################################################## Simple fully connected neural network ####################################################


class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Generate data for a task
def generate_task_data(func, num_samples, input_dim):
    x = np.random.randn(num_samples, input_dim)
    y = func(x)
    return torch.FloatTensor(x).to(device), torch.FloatTensor(y).to(device)


# Initialize scalers
x_scaler = StandardScaler()
y_scaler = StandardScaler()


def preprocess_data(train_x, train_y, test_x, test_y):

    # Fit scalers on training data
    train_x = x_scaler.fit_transform(train_x)
    train_y = y_scaler.fit_transform(train_y)

    # Transform test data
    test_x = x_scaler.transform(test_x)
    test_y = y_scaler.transform(test_y)
    return (
        torch.FloatTensor(train_x).to(device),
        torch.FloatTensor(train_y).to(device),
        torch.FloatTensor(test_x).to(device),
        torch.FloatTensor(test_y).to(device),
    )


input_dim = 5
output_dim = 5
func = generate_polynomial_functions(1)[0]
num_epochs = 30000

model = SimpleNet(input_dim, output_dim).to(device)
weight_decay = 1e-5  # L2 Regularization
optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)


# Generate training and testing data once (For Pre-training)
num_train_samples = 1000
num_test_samples = 1000
train_x, train_y = generate_task_data(func, num_train_samples, input_dim)
test_x, test_y = generate_task_data(func, num_test_samples, input_dim)

train_x, train_y, test_x, test_y = preprocess_data(train_x, train_y, test_x, test_y)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(train_x)
    loss = criterion(pred, train_y)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss = criterion(test_pred, test_y)
        print(
            f"Pretrain Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}"
        )


model_save_path = os.path.join(working_dir, "Nominal_Model.pth")
torch.save(model.state_dict(), model_save_path)
