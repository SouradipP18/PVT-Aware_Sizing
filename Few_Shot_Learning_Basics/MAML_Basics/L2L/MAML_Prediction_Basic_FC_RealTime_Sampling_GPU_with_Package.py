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
import learn2learn as l2l

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
        self.fc1 = nn.Linear(input_dim, 512)  # 256
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


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
output_dim = 3
num_inner_loop_steps = 20  # 5

# Hyper-parameters
num_tasks = (
    45  # 16+1 # More the tasks, better the generalization/evaluation score on new tasks
)
tasks_per_batch = 8  # 5
num_batches = 6  # 10 # 100 # 10000
num_support_samples = 200 # During training
num_query_samples = 500   # During training
num_epochs = 700  # 450 # 1200  # 2000  # 500  # 10000  # Number of Meta-Iterations

# Training functions:
functions = generate_polynomial_functions(num_tasks)
test_functions = generate_polynomial_functions(5)

num_support_test_samples = num_support_samples  # # Small since the available data would be small during evaluation
num_query_test_samples = num_query_samples  # Just for evaluation data (not counted since this info is not used to model during testing)


query_losses = []
support_losses = []

# Total number of samples (support + query)
total_samples = num_support_samples + num_query_samples

data_usage_counter = {}

# Main training loop
def train_maml(maml, functions, meta_optimizer, meta_scheduler, num_epochs, tasks_per_batch):
    for epoch in range(num_epochs):  # Each meta-iteration
        total_loss = 0
        total_tasks_epoch = []
        maml.train()
        for batch in range(num_batches):
            tasks = []  # Functions for this batch
            selected_funcs = np.random.choice(
                len(functions), tasks_per_batch, replace=False
            )
            print(
                f"Epoch {epoch+1}, Batch {batch+1}, Selected Task IDs: {selected_funcs}"
            )
            for func_id in selected_funcs:
                # Initialize the counter for the current func_id if not already present
                if func_id not in data_usage_counter:
                    data_usage_counter[func_id] = 0
                # Sampling Support And Query for this function i.e. task T
                x_total = np.random.randn(total_samples, input_dim)
                x_total = torch.FloatTensor(x_total)
                indices = np.random.permutation(total_samples)
                support_indices = indices[:num_support_samples]
                query_indices = indices[num_support_samples:]
                func = functions[func_id]
                # Update the counter with the number of samples used
                data_usage_counter[func_id] += total_samples
                y_total = func(x_total.cpu().numpy())
                y_total = torch.FloatTensor(y_total)
                task_support_x, task_support_y = (
                    x_total[support_indices],
                    y_total[support_indices],
                )
                task_query_x, task_query_y = (
                    x_total[query_indices],
                    y_total[query_indices],
                )
                task_support_x, task_support_y, task_query_x, task_query_y = (
                    preprocess_data(
                        task_support_x, task_support_y, task_query_x, task_query_y
                    )
                )
                tasks.append(
                    (task_support_x, task_support_y, task_query_x, task_query_y)
                )
                total_tasks_epoch.append(
                    (task_support_x, task_support_y, task_query_x, task_query_y)
                )
            
            # Training maml for this batch
            meta_loss = 0.0
            for support_x, support_y, query_x, query_y in tasks:
                task_learner = maml.clone().to(device)  # Clone the meta-learner
                # Inner loop: Adapt the model using the support set
                for _ in range(num_inner_loop_steps):
                    preds = task_learner(support_x)
                    task_loss = nn.MSELoss()(preds, support_y)
                    task_learner.adapt(task_loss)
                
                # Outer loop: Evaluate the adapted model on the query set
                query_preds = task_learner(query_x)
                query_loss = nn.MSELoss()(query_preds, query_y)
                meta_loss += query_loss

            # Meta-optimization step
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
            total_loss += meta_loss.item()
        meta_scheduler.step()
        avg_loss = total_loss / (tasks_per_batch * num_batches)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        query_losses.append(avg_loss)

        # Periodic evaluation
        if (epoch + 1) % 50 == 0:  # Evaluate every 50 epochs
            total_eval_loss = 0
            for (
                eval_support_x,
                eval_support_y,
                eval_query_x,
                eval_query_y,
            ) in total_tasks_epoch:
                adapted_model_eval = maml.clone().to(device)
                # Inner loop: Adapt the model using the eval_support set
                for _ in range(num_inner_loop_steps):
                    eval_preds = adapted_model_eval(eval_support_x)
                    eval_task_loss = nn.MSELoss()(eval_preds, eval_support_y)
                    adapted_model_eval.adapt(eval_task_loss)

                eval_query_pred = adapted_model_eval(eval_query_x)
                eval_loss = nn.MSELoss()(eval_query_pred, eval_query_y)
                total_eval_loss += eval_loss.item()
            print(total_eval_loss / len(total_tasks_epoch))

        # Add periodic validation
        # if (epoch + 1) % 10 == 0:
        #    eval_loss = evaluate_maml(maml, functions, num_samples=100)
        #    print(f"Evaluation Loss at Epoch {epoch+1}: {eval_loss:.4f}")


################### Evaluation of the learnt meta/starting parameters on 5 Unseen functions ########################

total_test_samples = num_support_samples + num_query_test_samples


def evaluate_maml(maml, test_functions):
    maml.eval()
    print()
    print(
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Evaluation Starts Here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    )
    total_loss = 0
    for test_func_idx, test_func in enumerate(test_functions):
        print()
        print(f"Evaluation Task {test_func_idx+1} :::::::::::::::::::")
        # Sampling Support And Query for this function i.e. task T
        test_x_total = np.random.randn(total_test_samples, input_dim)
        test_x_total = torch.FloatTensor(test_x_total)
        test_indices = np.random.permutation(total_test_samples)
        support_indices = test_indices[:num_support_samples]
        query_indices = test_indices[num_support_samples:]
        test_y_total = test_func(test_x_total.cpu().numpy())
        test_y_total = torch.FloatTensor(test_y_total)
        test_support_x, test_support_y = (
            test_x_total[support_indices],
            test_y_total[support_indices],
        )
        test_query_x, test_query_y = (
            test_x_total[query_indices],
            test_y_total[query_indices],
        )
        test_support_x, test_support_y, test_query_x, test_query_y = preprocess_data(
            test_support_x, test_support_y, test_query_x, test_query_y
        )
        # Adapt to the task
        eval_task_learner = maml.clone().to(device)
        for _ in range(num_inner_loop_steps):
            preds = eval_task_learner(test_support_x)
            eval_task_loss = nn.MSELoss()(preds, test_support_y)
            eval_task_learner.adapt(eval_task_loss)
        # Evaluate on query set
        with torch.no_grad():
            query_preds = eval_task_learner(test_query_x)
            loss = nn.MSELoss()(query_preds, test_query_y)
            total_loss += loss.item()
        print(f"  Query Loss: {loss.item():.4f}")
    return total_loss / len(test_functions)


# You can use a pre-trained model here and then the above algorithm would just fine-tune this model then. Also should save training time.
model = SimpleNet(input_dim, output_dim).to(device)

maml = l2l.algorithms.MAML(model, lr=0.001, first_order=False).to(device)
maml_optimizer = optim.Adam(maml.parameters())
maml_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(maml_optimizer, T_0=100)

train_maml(maml, functions, meta_optimizer=maml_optimizer, meta_scheduler=maml_scheduler, num_epochs=num_epochs, tasks_per_batch=tasks_per_batch)
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


indices = list(data_usage_counter.keys())
data_counts = list(data_usage_counter.values())

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(indices, data_counts, color='skyblue', edgecolor='black')
plt.xlabel("Function ID")
plt.ylabel("Total Data Used")
plt.title("Data Usage per Function")
plt.xticks(indices)  # Ensure proper labeling of function IDs
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()