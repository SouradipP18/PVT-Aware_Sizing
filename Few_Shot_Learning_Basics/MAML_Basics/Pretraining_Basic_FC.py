import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.cuda as cuda
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if cuda.is_available() else "cpu")


# Generate 17 similar functions
def generate_functions(num_functions, input_dim=5, output_dim=5):
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


######################################################## Simple fully connected neural network ####################################################


class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, output_dim)

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
func = generate_functions(1, input_dim, output_dim)[0]
num_epochs = 1000

model = SimpleNet(input_dim, output_dim).to(device)
weight_decay = 1e-5  # L2 Regularization
optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)


# Generate training and testing data once
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
