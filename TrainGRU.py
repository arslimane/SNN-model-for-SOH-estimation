import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools
import struct
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm, trange
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torchvision.models as models
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import pynvml

dtype = torch.float
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

loss2 = nn.MSELoss()
loss = nn.L1Loss()


def measure_accuracy_gru(model, dataloader):
    """Compute MAE, MSE, RMSE, STD, 95% CI, and average inference time for GRU model."""
    t = []
    errors = []

    model.eval()
    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            t1 = time.time()
            outputs = model(data)
            t2 = time.time()
            t.append(t2 - t1)

            batch_errors = torch.abs(outputs - targets)
            errors.append(batch_errors.cpu().numpy())

    errors = np.concatenate(errors).flatten()
    mae = np.mean(errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    std = np.std(errors)
    ci95 = 1.96 * std / np.sqrt(len(errors))
    avg_time = np.mean(t)

    return mae, mse, rmse, std, ci95, avg_time


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class GRU_CNN(nn.Module):
    def __init__(self, stepin, stpout):
        super(GRU_CNN, self).__init__()

        self.conv_block = nn.Sequential(
            Reshape((3, stepin * 1500)),  # From (batch, 37500, 3) ➜ (batch, 3, 37500)
            nn.Conv1d(3, 64, kernel_size=32, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(4),
        )

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(64 * 4, 64)

        self.gru = nn.GRU(3, 256, batch_first=True)
        self.dense2 = nn.Linear(256, 64)

        self.concat = nn.Linear(128, stpout)

    def forward(self, input_stream):
        x1 = self.conv_block(input_stream)
        x1 = self.flatten(x1)
        x1 = self.dense1(x1)
        _, x2 = self.gru(input_stream)
        x2 = x2.squeeze(0)
        x2 = self.dense2(x2)

        combined = torch.cat((x1, x2), dim=1)
        output = self.concat(combined)

        return output


def Test_GRUModel(Steps_in, Steps_out):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    for stpsin, stpout in tqdm(zip(Steps_in, Steps_out)):
        # -------------------------
        # Load training and test data
        # -------------------------
        Xtrain = np.load(f"Data/Xtrain_normilized_{stpsin}_{stpout}.npz")["Data"].astype(np.float32)
        Ytrain = np.load(f"Data/Ytrain_{stpsin}_{stpout}.npz")["y"].astype(np.float32)
        Xtest = np.load(f"Data/Xtest_normilized_{stpsin}_{stpout}.npz")["Data"].astype(np.float32)
        Ytest = np.load(f"Data/Ytest_{stpsin}_{stpout}.npz")["y"].astype(np.float32)

        Ytrain = np.reshape(Ytrain, (len(Ytrain), stpout))
        Ytest = np.reshape(Ytest, (len(Ytest), stpout))

        train_dataset = TensorDataset(torch.tensor(Xtrain), torch.tensor(Ytrain))
        test_dataset = TensorDataset(torch.tensor(Xtest), torch.tensor(Ytest))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # -------------------------
        # Initialize model, loss, optimizer
        # -------------------------
        model = GRU_CNN(stpsin, stpout).to(device)
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

        # -------------------------
        # Training loop
        # -------------------------
        num_epochs = 1
        for epoch in range(num_epochs):
            model.train()
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss_val = loss_fn(outputs, targets)
                loss_val.backward()
                optimizer.step()

        # Save model
        torch.save(model.state_dict(), f"Models/modelGRU_weights_{stpsin}_{stpout}.pth")
        # -------------------------
        # Cleanup
        # -------------------------
        del model, optimizer, train_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()



Steps_in = [1,10,20, 25, 25, 25, 25]
Steps_out = [1,5,10, 25, 50, 100, 200]

Test_GRUModel(Steps_in, Steps_out)
