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
from collections import OrderedDict


working_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(working_dir)

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Initialize scalers
x_scaler = StandardScaler()
y_scaler = StandardScaler()

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
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# MAML algorithm
class MAML:
    def __init__(self, model, inner_lr, num_inner_steps):
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = optim.Adam(self.model.parameters(), weight_decay=1e-5)
        self.meta_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.meta_optimizer, T_0=100
        )

    def update_params(self, params, loss, first_order=True):
        name_list, tensor_list = zip(*params.items())
        grads = torch.autograd.grad(
            loss, tensor_list, create_graph=not first_order, allow_unused=True
        )
        updated_params = OrderedDict()
        for name, param, grad in zip(name_list, tensor_list, grads):
            if grad is not None:
                updated_params[name] = param - self.inner_lr * grad
            else:
                updated_params[name] = param
        return updated_params

    def inner_loop(self, support_x, support_y):
        present_model = self.clone_model().to(device)
        updated_params = OrderedDict(present_model.named_parameters())

        for _ in range(self.num_inner_steps):
            # present_model.load_state_dict(updated_params)
            for name, param in present_model.named_parameters():
                param.data = updated_params[name].data
            pred = present_model(support_x)
            loss = nn.MSELoss()(pred, support_y)
            updated_params = self.update_params(
                OrderedDict(present_model.named_parameters()), loss
            )
            print(f"    Inner Step {_+1}, Support Loss: {loss.item():.4f}")
        return updated_params

    def outer_step(self, tasks):
        meta_loss = 0
        original_params = OrderedDict(self.model.named_parameters())
        for task_idx, (support_x, support_y, query_x, query_y) in enumerate(tasks):
            print(f"  Running Task {task_idx+1}")
            task_adapted_params_dict = self.inner_loop(support_x, support_y)
            # Temporarily update model parameters
            for name, param in self.model.named_parameters():
                param.data = task_adapted_params_dict[name].data
            query_pred = self.model(query_x)
            query_loss = nn.MSELoss()(query_pred, query_y)
            meta_loss += query_loss
            print(f"  Task {task_idx+1}, Query Loss: {query_loss.item():.4f}")
            # Restore original parameters
            for name, param in self.model.named_parameters():
                param.data = original_params[name].data

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.norm().item()}")
        #     else:
        #         print(f"No gradient for {name}")

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        self.meta_scheduler.step()
        return meta_loss.item()

    def clone_model(self):
        return copy.deepcopy(self.model).to(device)


# Generate data for a task
def generate_task_data(func, num_samples, input_dim):
    x = np.random.randn(num_samples, input_dim)
    y = func(x)
    return torch.FloatTensor(x).to(device), torch.FloatTensor(y).to(device)


def preprocess_data(train_x, train_y, test_x, test_y):
    # Move tensors to CPU and convert to NumPy arrays
    train_x = train_x.cpu().numpy()
    train_y = train_y.cpu().numpy()
    test_x = test_x.cpu().numpy()
    test_y = test_y.cpu().numpy()

    # Fit scalers on training data
    train_x = x_scaler.fit_transform(train_x)
    train_y = y_scaler.fit_transform(train_y)

    # Transform test data
    test_x = x_scaler.transform(test_x)
    test_y = y_scaler.transform(test_y)

    # Convert back to PyTorch tensors and move to the GPU
    return (
        torch.FloatTensor(train_x).to(device),
        torch.FloatTensor(train_y).to(device),
        torch.FloatTensor(test_x).to(device),
        torch.FloatTensor(test_y).to(device),
    )


input_dim = 5
output_dim = 5
num_inner_loop_steps = 10  # 5

# Hyper-parameters
num_tasks = (
    45  # 16+1 # More the tasks, better the generalization/evaluation score on new tasks
)
tasks_per_batch = 5  # 5
num_batches = 10  # 10 # 100 # 10000
num_support_samples = (
    200  # = Number of "shots". "Few Shots" during training and inference/evaluation
)
num_query_samples = 500  # During training
num_epochs = 1200  # 2000  # 500  # 10000  # Number of Meta-Iterations

# functions = generate_simple_functions(num_tasks, input_dim, output_dim)
functions = generate_polynomial_functions(num_tasks)

query_losses = []
support_losses = []

# Pre-generate task data to keep a count on the amount of training data needed
task_data = {}
tasks_eval = []
for func_id, func in enumerate(functions):
    task_support_x, task_support_y = generate_task_data(
        func, num_support_samples, input_dim
    )
    task_query_x, task_query_y = generate_task_data(func, num_query_samples, input_dim)
    task_support_x, task_support_y, task_query_x, task_query_y = preprocess_data(
        task_support_x, task_support_y, task_query_x, task_query_y
    )
    task_data[func_id] = (task_support_x, task_support_y, task_query_x, task_query_y)
    tasks_eval.append(task_data[func_id])


# Main training loop
def train_maml(maml, functions, num_epochs, tasks_per_batch):
    for epoch in range(num_epochs):  # Each meta-iteration
        total_loss = 0
        maml.model.train()
        for batch in range(num_batches):
            tasks = []
            selected_funcs = np.random.choice(
                len(functions), tasks_per_batch, replace=False
            )
            print(
                f"Epoch {epoch+1}, Batch {batch+1}, Selected Task IDs: {selected_funcs}"
            )
            for func_id in selected_funcs:
                tasks.append(task_data[func_id])
            loss = maml.outer_step(tasks)
            total_loss += loss
        avg_loss = total_loss / (tasks_per_batch * num_batches)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        query_losses.append(avg_loss)

        # Periodic evaluation
        if (epoch + 1) % 50 == 0:  # Evaluate every 50 epochs
            total_eval_loss = 0
            for func_idx_eval, func_eval in enumerate(functions):
                eval_support_x, eval_support_y, eval_query_x, eval_query_y = tasks_eval[
                    func_idx_eval
                ]

                adapted_model_eval = maml.clone_model().to(device)
                adapted_params_eval_dict = maml.inner_loop(
                    eval_support_x, eval_support_y
                )
                for name, param in adapted_model_eval.named_parameters():
                    param.data = adapted_params_eval_dict[name].data

                eval_query_pred = adapted_model_eval(eval_query_x)
                eval_loss = nn.MSELoss()(eval_query_pred, eval_query_y)
                total_eval_loss += eval_loss.item()
            print(total_eval_loss / len(functions))

        # Add periodic validation
        # if (epoch + 1) % 10 == 0:
        #    eval_loss = evaluate_maml(maml, functions, num_samples=100)
        #    print(f"Evaluation Loss at Epoch {epoch+1}: {eval_loss:.4f}")


################### Evaluation of the learnt meta/starting parameters on 5 Unseen functions ########################

# test_functions = generate_simple_functions(5, input_dim, output_dim)
test_functions = generate_polynomial_functions(5)
num_support_test_samples = num_support_samples  # # Small since the available data would be small during evaluation
num_query_test_samples = 400  # Just for evaluation data (not counted since this info is not used to model during testing)


def evaluate_maml(maml, test_functions):
    maml.model.eval()
    print()
    print(
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Evaluation Starts Here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    )
    total_loss = 0
    for func_idx, func in enumerate(test_functions):
        print()
        print(f"Evaluation Task {func_idx+1} :::::::::::::::::::")
        support_x, support_y = generate_task_data(
            func, num_support_test_samples, input_dim
        )
        query_x, query_y = generate_task_data(func, num_query_test_samples, input_dim)

        # Adapt to the task
        adapted_params_dict = maml.inner_loop(support_x, support_y)

        # Evaluate on query set
        with torch.no_grad():
            adapted_model = maml.clone_model().to(device)
            adapted_model.load_state_dict(adapted_params_dict)
            query_pred = adapted_model(query_x)
            loss = nn.MSELoss()(query_pred, query_y)
            total_loss += loss.item()
        print(f"  Query Loss: {loss.item():.4f}")
    return total_loss / len(test_functions)


# You can use a pre-trained model here and then the above algorithm would just fine-tune this model then. Also should save training time.
model = SimpleNet(input_dim, output_dim).to(device)

# # Loading the Pre-Trained Weights from the Nominal
# model_save_path = os.path.join(working_dir, "Nominal_Model.pth")
# model.load_state_dict(torch.load(model_save_path))

maml = MAML(model, inner_lr=0.001, num_inner_steps=num_inner_loop_steps)
train_maml(maml, functions, num_epochs=num_epochs, tasks_per_batch=tasks_per_batch)
eval_loss = evaluate_maml(maml, test_functions)
print(f"Evaluation Loss: {eval_loss}")

# Make sure that when you use this MAML class, you provide tasks in the correct format (support_x, support_y, query_x, query_y) for the outer_step method.
# Consider adding a method for evaluation or prediction using the meta-learned model.


# Plotting the Query losses v/s epochs
plt.figure(figsize=(8, 4))  # You can adjust the size of the figure
plt.plot(
    list(range(1, len(query_losses) + 1)),
    query_losses,
    marker="o",
    linestyle="-",
    color="b",
)  # Line plot with blue line and circle markers

# Adding title and labels
plt.title("Plot of List Values vs. Indices")
plt.xlabel("Index")
plt.ylabel("Value")

# Show grid
plt.grid(True)

# Display the plot
plt.show()

print(f"Memory address of the model: {hex(id(maml.model))}")
