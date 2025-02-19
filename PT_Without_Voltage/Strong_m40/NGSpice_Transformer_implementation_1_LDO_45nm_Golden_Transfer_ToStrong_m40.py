import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math
import numpy as np
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math
import torch.utils.data as data
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
)
from sklearn.impute import SimpleImputer
import pickle
import os
import pandas as pd
import sys

working_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(working_dir)


def round_scientific(x):
    """Round the number but keep scientific notation"""
    return pd.Series(x).apply(lambda y: float(f"{y:.1e}"))


##################################################################################################################################################
############################################################## Model Parameters ##################################################################
##################################################################################################################################################

d_model = 16  # Dimension of the model
nhead = 4  # Number of heads in the multiheadattention models
num_decoder_layers = 6  # Number of sub-encoder-layers in the decoder

#################################################################################################################################################
############################################################## Input Data Type ####################################################################
###################################################################################################################################################

input_incomplete_sequence_length = 6
number_perf = 7
total_sequence_length = input_incomplete_sequence_length + number_perf
# total_sequences_dataset_size = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#############################################################
df_45nm_train_val = pd.read_excel(os.path.join(working_dir, "Golden_train_data.xlsx"))
df_45nm_test = pd.read_excel(os.path.join(working_dir, "Golden_test_data.xlsx"))

params_header = [
    "wn_tail",
    "wp_input",
    "wp_load",
    "wp_tail",
    "c_miller",
    "cl",
]

present_order_header = params_header + [
    "IQ",
    "Efficiency",
    "Dropout",
    "Phase_Margin",
    "PSRR",
    "Line_Reg_Percent",
    "Load_Reg_Percent",
]

df_45nm_train_val.columns = present_order_header
df_45nm_test.columns = present_order_header

print(df_45nm_train_val.head())
print(df_45nm_test.head())

# Reordering::::::::::::::::::::::::::::::::::::::::::::::::

header_reordered = params_header + [
    "IQ",
    "Efficiency",
    "Dropout",
    "Phase_Margin",
    "PSRR",
    "Line_Reg_Percent",
    "Load_Reg_Percent",
]

df_45nm_train_val = df_45nm_train_val[header_reordered]
df_45nm_test = df_45nm_test[header_reordered]

print(df_45nm_train_val.head())
print(df_45nm_test.head())

df_cleaned_sampled_train_val = df_45nm_train_val.sample(n=800, random_state=4)
array_cleaned_train = df_cleaned_sampled_train_val.to_numpy()

array_cleaned_test = df_45nm_test.to_numpy()

train_val_size = array_cleaned_train.shape[0]
print("Total 45nm Train-Val Dataset Size:", train_val_size)

"""
new_df["GBW"] = np.log(new_df["GBW"])
new_df["IRN_density"] = np.log(new_df["IRN_density"])
new_df["Slew_Rate"] = np.log(np.abs(new_df["Slew_Rate"]))
new_df["Rise_Time"] = np.log(np.abs(new_df["Rise_Time"]))
new_df["Fall_Time"] = np.log(np.abs(new_df["Fall_Time"]))
"""
########################################## Replace this Part for Circuit Data ########################################################

total_train_val_xs = array_cleaned_train[:, :input_incomplete_sequence_length]
total_train_val_ys = array_cleaned_train[:, input_incomplete_sequence_length:]
total_test_xs = array_cleaned_test[:, :input_incomplete_sequence_length]
total_test_ys = array_cleaned_test[:, input_incomplete_sequence_length:]

# Normalization
# Create a StandardScaler object
scaler_xs = StandardScaler()
scaler_ys = StandardScaler()

# Based on the DP-based optimal path
scaler1 = StandardScaler()
scaler3 = StandardScaler()
scaler4 = StandardScaler()
scaler2 = QuantileTransformer(n_quantiles=200, output_distribution="uniform")
scaler6 = StandardScaler()
scaler7 = StandardScaler()
scaler5 = StandardScaler()


####################################################################################

# Fit the scaler to the data and transform it for xs
normalized_total_train_val_xs = scaler_xs.fit_transform(total_train_val_xs)
normalized_total_test_xs = scaler_xs.transform(total_test_xs)

# Fit the scaler to the data and transform it for ys
# normalized_total_train_test_ys = scaler_ys.fit_transform(total_train_test_ys)

normalized_total_train_val_ys = np.column_stack(
    [
        scaler1.fit_transform(total_train_val_ys[:, 0].reshape(-1, 1)).flatten(),
        scaler3.fit_transform(total_train_val_ys[:, 2].reshape(-1, 1)).flatten(),
        scaler4.fit_transform(total_train_val_ys[:, 3].reshape(-1, 1)).flatten(),
        scaler2.fit_transform(total_train_val_ys[:, 1].reshape(-1, 1)).flatten(),
        scaler6.fit_transform(total_train_val_ys[:, 5].reshape(-1, 1)).flatten(),
        scaler7.fit_transform(total_train_val_ys[:, 6].reshape(-1, 1)).flatten(),
        scaler5.fit_transform(total_train_val_ys[:, 4].reshape(-1, 1)).flatten(),
    ]
)
normalized_total_test_ys = np.column_stack(
    [
        scaler1.transform(total_test_ys[:, 0].reshape(-1, 1)).flatten(),
        scaler3.transform(total_test_ys[:, 2].reshape(-1, 1)).flatten(),
        scaler4.transform(total_test_ys[:, 3].reshape(-1, 1)).flatten(),
        scaler2.transform(total_test_ys[:, 1].reshape(-1, 1)).flatten(),
        scaler6.transform(total_test_ys[:, 5].reshape(-1, 1)).flatten(),
        scaler7.transform(total_test_ys[:, 6].reshape(-1, 1)).flatten(),
        scaler5.transform(total_test_ys[:, 4].reshape(-1, 1)).flatten(),
    ]
)


tensor_train_val_xs = torch.tensor(normalized_total_train_val_xs, dtype=torch.float32)
tensor_train_val_ys = torch.tensor(normalized_total_train_val_ys, dtype=torch.float32)
tensor_test_xs = torch.tensor(normalized_total_test_xs, dtype=torch.float32)
tensor_test_ys = torch.tensor(normalized_total_test_ys, dtype=torch.float32)


# Add a third dimension to the tensors using unsqueeze
# The -1 argument adds a dimension at the last position
total_train_val_xs = tensor_train_val_xs.unsqueeze(-1).to(device)
total_train_val_xs = torch.nn.utils.rnn.pad_sequence(
    total_train_val_xs, batch_first=True, padding_value=-torch.inf
)
total_train_val_ys = tensor_train_val_ys.unsqueeze(-1).to(device)

total_test_xs = tensor_test_xs.unsqueeze(-1).to(device)
total_test_xs = torch.nn.utils.rnn.pad_sequence(
    total_test_xs, batch_first=True, padding_value=-torch.inf
)
total_test_ys = tensor_test_ys.unsqueeze(-1).to(device)

###########################################################################################################################################

# Define the path to the pickle file
pickle_path = "/home/local/eda15/SAIT_Collaboration/Transformer-Learning-in-VLSI/scalers_circuits_NG.pkl"


# Function to load or initialize scalers dictionary
def load_or_initialize_scalers(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        return {}


# Function to save scalers dictionary
def save_scalers(scalers, path):
    with open(path, "wb") as f:
        pickle.dump(scalers, f)


# Main function to add a new scaler
def add_scalar(index, scaler):
    scalers = load_or_initialize_scalers(pickle_path)
    scalers[index] = scaler
    save_scalers(scalers, pickle_path)


add_scalar("xs2", scaler_xs)
add_scalar("ys2", scaler_ys)

###########################################################################################################################################


class CustomDataset(data.Dataset):
    def __init__(self, xs, ys):
        ones_tensor = torch.ones(xs.shape[0], 1, 1, device=xs.device) * (-1000)
        self.data = torch.cat([xs, ones_tensor, ys], 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


train_val_dataset = CustomDataset(total_train_val_xs, total_train_val_ys)
test_dataset = CustomDataset(total_test_xs, total_test_ys)
train_val_size = len(train_val_dataset)
print(train_val_size)
# exit()
###################################################### 80-20 Train:Val Split ########################################################
val_size = int(0.2 * train_val_size)
train_size = train_val_size - val_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

train_dataset_loader, val_dataset_loader, test_dataset_loader = (
    DataLoader(train_dataset, batch_size=64, shuffle=True),
    DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False),
    DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False),
)


temp_dir = os.path.join(working_dir, ".tmp")
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


class EarlyStopping:
    def __init__(self, patience=300, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.epochs_no_improve = 0
        self.stop = False
        self.save_path = os.path.join(temp_dir, "temp_model.pth")

    def __call__(self, val_loss):
        if self.best_loss == None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.stop = True


early_stopping = EarlyStopping()


class TransformerDecoderModel(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers):
        super(TransformerDecoderModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Sequential(
            nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

        decoder_layer = TransformerDecoderLayer(d_model, nhead, 4 * d_model)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        # self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        self.pos_encoder = nn.Embedding(
            total_sequence_length + 1, d_model
        )  # Added the <START> sequence position
        # self.type_encoder = nn.Embedding(10, d_model)
        self.type_encoder = nn.Embedding(
            3, d_model
        )  # Row 0 for padding,rows 1 and 2 for integers "1" and "2"

    def forward(self, tgt, memory, tgt_mask):
        tgt = self.embedding(tgt)
        # To understand what type the numbers are: parameters or performance numbers
        # type_vector = [1] * input_incomplete_sequence_length + [2]
        type_vector = [1] * input_incomplete_sequence_length + [0]
        for idx in range(tgt.shape[1] - input_incomplete_sequence_length - 1):
            type_vector += [2]
        type_vector = (
            torch.tensor(type_vector)
            .reshape(1, -1)
            .repeat(tgt.shape[0], 1)
            .to(tgt.device)
        )
        type_emb = self.type_encoder(type_vector).to(tgt.device)
        pos_encoder_src = (
            torch.unsqueeze(
                self.pos_encoder(torch.arange(tgt.shape[1], device=device).long()), 0
            )
            .repeat(tgt.shape[0], 1, 1)
            .to(tgt.device)
        )
        tgt = tgt + pos_encoder_src + type_emb
        tgt = torch.transpose(tgt, 1, 0).to(device)
        if memory is None:
            # If memory is None, we assume no encoder context, and just pass a zero tensor
            memory = torch.zeros(
                (tgt.size(0), tgt.size(1), self.d_model), device=tgt.device
            )
        tgt_mask = tgt_mask.to(tgt.device)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1).to(device)
    return mask


model = TransformerDecoderModel(d_model, nhead, num_decoder_layers)
model.load_state_dict(
    torch.load(
        os.path.join(working_dir, "NGSpice_Model_LDO_Cadence_7_Specs_Nominal_27_modif")
    )
)
model.to(device)
optimizer = optim.Adam(model.parameters())
# loss_fn = nn.MSELoss()
loss_fn = nn.MSELoss()
valid_loss = None
# num_epochs = 10000  # Number of epochs to train
# num_epochs = 500

num_epochs = 1500  # 900
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

test_l2 = None
test_r2 = None
val_r2 = None
for epoch in range(num_epochs):

    total_loss = 0
    pbar = tqdm(train_dataset_loader)
    model.train()
    for tgt in pbar:
        tgt = tgt.to(device).float()
        tgt_input = tgt[:, :-1]
        targets = tgt[:, 1:].contiguous()
        src_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        outputs = model(tgt_input, None, src_mask)
        pred_val = torch.transpose(outputs[-number_perf:, :, :], 1, 0)
        true_val = targets[:, -number_perf:].reshape(pred_val.shape)
        loss = loss_fn(pred_val, true_val)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss = loss.item()
        pbar.set_description_str(
            f"Episode: {epoch}: train l2: {train_loss}, test l2: {test_l2}, test r2: {test_r2}"
        )

    pbar = tqdm(test_dataset_loader)
    model.eval()
    with torch.no_grad():
        for tgt in pbar:
            tgt = tgt.to(device).float()
            tgt_input = tgt[:, :-1]
            targets = tgt[:, 1:].contiguous()
            src_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            outputs = model(tgt_input, None, src_mask)
            pred_val = torch.transpose(outputs[-number_perf:, :, :], 1, 0)
            true_val = targets[:, -number_perf:].reshape(pred_val.shape)
            test_l2 = loss_fn(pred_val, true_val)
            test_l2 = test_l2.item()
            test_r2 = r2_score(
                pred_val.detach().cpu().numpy()[:, :, 0],
                true_val.detach().cpu().numpy()[:, :, 0],
            )
    total_val_loss = 0.0
    total_val_count = 0
    pbar = tqdm(val_dataset_loader)
    with torch.no_grad():
        for tgt in pbar:
            tgt = tgt.to(device).float()
            tgt_input = tgt[:, :-1]
            targets = tgt[:, 1:].contiguous()
            src_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            outputs = model(tgt_input, None, src_mask)
            pred_val = torch.transpose(outputs[-number_perf:, :, :], 1, 0)
            true_val = targets[:, -number_perf:].reshape(pred_val.shape)
            val_l2 = loss_fn(pred_val, true_val)
            total_val_loss += val_l2.item()
            total_val_count += 1
            val_r2 = r2_score(
                pred_val.detach().cpu().numpy()[:, :, 0],
                true_val.detach().cpu().numpy()[:, :, 0],
            )
    avg_val_loss = total_val_loss / total_val_count
    early_stopping(avg_val_loss)
    if early_stopping.stop:
        print("Early stopping triggered")
        break

model.load_state_dict(torch.load(early_stopping.save_path))
shutil.rmtree(temp_dir)
model.eval()
model.to(device)

pbar = tqdm(test_dataset_loader)
with torch.no_grad():
    for tgt in pbar:
        tgt = tgt.to(device).float()
        tgt_input = tgt[:, : 1 + torch.where(tgt == -1000)[1][0].item()].contiguous()
        for _ in range(number_perf):
            src_mask = torch.triu(
                torch.ones(tgt_input.shape[1], tgt_input.shape[1]) * float("-inf"),
                diagonal=1,
            )
            outputs = model(tgt_input, None, src_mask)
            outputs = outputs[-1, :].reshape(-1, 1, 1)
            tgt_input = torch.cat([tgt_input, outputs], 1)
        tgt_input = torch.transpose(tgt_input, 1, 0)
        tgt_input = tgt_input[-number_perf:]
        pred_val = torch.transpose(tgt_input[-number_perf:], 1, 0)[:, :, 0]
        true_val = tgt[:, -number_perf:].reshape(pred_val.shape)
        loss = nn.functional.mse_loss(pred_val, true_val)
        print(f"Test Prediction loss: {loss.item()}")
        # Total R2 computation:::
        # You need to move the tensors to CPU and convert them to NumPy arrays
        pred_val_np = pred_val.detach().cpu().numpy()
        true_val_np = true_val.detach().cpu().numpy()
        r2 = r2_score(true_val_np, pred_val_np)
        print(f"R-squared score: {r2}")
