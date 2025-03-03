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


def generate_polynomial_functions(num_functions, input_dim=5, output_dim=3):
    polyfunctions = []
    for _ in range(num_functions):
        coeffs = {
            "linear": np.random.randn(output_dim, input_dim) * 0.1,
            "quadratic": np.random.randn(output_dim, input_dim) * 0.1,
            "cubic": np.random.randn(output_dim, input_dim) * 0.1,
        }
        bias = np.random.randn(output_dim) * 0.1

        def poly_func(x, coeff=coeffs, b=bias):
            y = np.zeros((x.shape[0], output_dim))
            for i in range(output_dim):
                # Linear term
                y[:, i] += np.dot(x, coeff["linear"][i, :])
                # Quadratic term
                y[:, i] += np.dot(np.power(x, 2), coeff["quadratic"][i, :])
                # Cubic term
                y[:, i] += np.dot(np.power(x, 3), coeff["cubic"][i, :])
                # Add bias
                y[:, i] += b[i]

            return y

        polyfunctions.append(poly_func)

    return polyfunctions


######################################################## Simple fully connected neural network ####################################################


class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


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
output_dim = 3
func = generate_polynomial_functions(1)[0]
num_epochs = 3000

model = SimpleNet(input_dim, output_dim).to(device)
weight_decay = 1e-5  # L2 Regularization
optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)


# Generate training and testing data once (For Pre-training)
num_train_samples = 200
num_test_samples = 500
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
