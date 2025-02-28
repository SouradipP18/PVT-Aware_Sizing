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
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# MAML algorithm
class MAML:
    def __init__(self, model, inner_lr, num_inner_steps):
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps  # Number of shots
        self.meta_optimizer = optim.Adam(self.model.parameters())
        self.meta_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.meta_optimizer, T_0=50
        )

    def inner_loop(self, support_x, support_y):
        present_model = self.clone_model()
        inner_loop_optimizer = optim.Adam(present_model.parameters(), self.inner_lr)
        for _ in range(self.num_inner_steps):
            # Using the current/latest task-specific params for the ongoing few-shot pass, not the self.model (meta-parameters)
            pred = present_model(support_x)
            loss = nn.MSELoss()(pred, support_y)

            # Also other optimizers like SGDS, Adam can be used here
            # grads = torch.autograd.grad(
            #     loss, present_model.parameters(), create_graph=True
            # )
            # # Now update the present model with the new params
            # for p, g in zip(present_model.parameters(), grads):
            #     p.data.sub_(self.inner_lr * g)
            inner_loop_optimizer.zero_grad()
            loss.backward(create_graph=True)
            inner_loop_optimizer.step()

            print(f"    Inner Step {_+1}, Support Loss: {loss.item():.4f}")
        return list(present_model.parameters())  # Last Updated Params

    def outer_step(self, tasks):
        outer_loss = 0
        for task_idx, (support_x, support_y, query_x, query_y) in enumerate(tasks):
            print(f"  Running Task {task_idx+1}")
            adapted_params = self.inner_loop(support_x, support_y)

            # Use adapted params to compute loss on query set
            with torch.set_grad_enabled(self.model.training):
                adapted_model = self.clone_model()
                adapted_model.load_state_dict(
                    {
                        name: param
                        for name, param in zip(
                            self.model.state_dict().keys(), adapted_params
                        )
                    }
                )
                query_pred = adapted_model(query_x)
                query_loss = nn.MSELoss()(query_pred, query_y)
                outer_loss += query_loss
            print(f"  Task {task_idx+1}, Query Loss: {query_loss.item():.4f}")
        avg_outer_loss = outer_loss / (tasks_per_batch * len(tasks))  # Per function
        self.meta_optimizer.zero_grad()
        avg_outer_loss.backward()
        self.meta_optimizer.step()
        self.meta_scheduler.step()
        return avg_outer_loss.item()

    def clone_model(self):
        # clone = copy.deepcopy(self.model)
        # clone.load_state_dict(self.model.state_dict())
        # return clone
        return copy.deepcopy(self.model).to(device)


# Generate data for a task
def generate_task_data(func, num_samples, input_dim):
    x = np.random.randn(num_samples, input_dim)
    y = func(x)
    return torch.FloatTensor(x).to(device), torch.FloatTensor(y).to(device)


input_dim = 5
output_dim = 5
num_shots = 2  # 5

# Hyper-parameters
num_tasks = 45  # 16+1
tasks_per_batch = 5
num_batches = 10  # 100 # 10000
num_support_samples = 10  # "Few Shots" during training and inference/evaluation
num_query_samples = 20  # During training
num_epochs = 100  # 10000  # Number of Meta-Iterations

functions = generate_functions(num_tasks, input_dim, output_dim)

query_losses = []
support_losses = []


# Main training loop
def train_maml(maml, functions, num_epochs, tasks_per_batch):
    for epoch in range(num_epochs):  # Each meta-iteration
        total_loss = 0
        for batch in range(num_batches):
            tasks = []
            selected_funcs = np.random.choice(
                len(functions), tasks_per_batch, replace=False
            )
            print(
                f"Epoch {epoch+1}, Batch {batch+1}, Selected Task IDs: {selected_funcs}"
            )
            for func_id in selected_funcs:
                func = functions[func_id]
                support_x, support_y = generate_task_data(
                    func, num_support_samples, input_dim
                )
                query_x, query_y = generate_task_data(
                    func, num_query_samples, input_dim
                )
                tasks.append((support_x, support_y, query_x, query_y))
            loss = maml.outer_step(tasks)
            total_loss += loss
        avg_loss = total_loss / (tasks_per_batch * num_batches)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        query_losses.append(avg_loss)

        # Add periodic validation
        # if (epoch + 1) % 10 == 0:
        #    eval_loss = evaluate_maml(maml, functions, num_samples=100)
        #    print(f"Evaluation Loss at Epoch {epoch+1}: {eval_loss:.4f}")


test_functions = generate_functions(
    5, input_dim, output_dim
)  # Evaluation of the learnt meta/starting parameters on 5 Unseen functions
num_support_test_samples = num_support_samples
num_query_test_samples = (
    20  # Small since the available data would be small during evaluation
)


def evaluate_maml(maml, test_functions):
    maml.model.eval()
    print()
    print(
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Evaluation Starts Here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    )
    print()
    total_loss = 0
    for func_idx, func in enumerate(test_functions):
        support_x, support_y = generate_task_data(
            func, num_support_test_samples, input_dim
        )
        query_x, query_y = generate_task_data(func, num_query_test_samples, input_dim)

        # Adapt to the task
        adapted_params = maml.inner_loop(support_x, support_y)

        # Evaluate on query set
        with torch.no_grad():
            adapted_model = maml.clone_model().to(device)
            adapted_model.load_state_dict(
                {
                    name: param
                    for name, param in zip(
                        adapted_model.state_dict().keys(), adapted_params
                    )
                }
            )
            query_pred = adapted_model(query_x)
            loss = nn.MSELoss()(query_pred, query_y)
            total_loss += loss.item()
        print(f"  Evaluation Task {func_idx+1}, Loss: {loss.item():.4f}")
    return total_loss / len(functions)


# You can use a pre-trained model here and then the above algorithm would just fine-tune this model then. Also should save training time.
model = SimpleNet(input_dim, output_dim).to(device)
maml = MAML(model, inner_lr=0.01, num_inner_steps=num_shots)
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
