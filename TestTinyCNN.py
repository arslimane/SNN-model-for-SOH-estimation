import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import gc
import time
from torch.utils.data import TensorDataset
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
class EmbeddedCNN_SOH(nn.Module):
    """
    Lightweight 1D CNN for SOH estimation based on the paper REIL-UConn “Rapid SOH Estimation from Short Pulses”
    Input: (batch, 1500 * stepin, 3)
    Output: (batch, stpout)
    """
    def __init__(self, stepin, stpout, hidden_channels=16):
        super().__init__()
        self.stepin = stepin
        self.stpout = stpout

        # -------------------------
        # CNN layers
        # -------------------------
        self.front = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=hidden_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.mid = nn.Sequential(
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels*2, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(hidden_channels*2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)   # global pooling -> (batch, channels, 1)
        )

        # -------------------------
        # Fully connected head
        # -------------------------
        self.head = nn.Sequential(
            nn.Flatten(),                  # -> (batch, channels)
            nn.Linear(hidden_channels*2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.stpout)     # output size = stpout
        )

    def forward(self, x):
        # x: (batch, time=1500*stepin, features=3)
        x = x.permute(0, 2, 1)            # -> (batch, features, time)
        x = self.front(x)
        x = self.mid(x)                   # -> (batch, channels, 1)
        x = x.squeeze(-1)                 # -> (batch, channels)
        out = self.head(x)                # -> (batch, stpout)
        return out

# -------------------------
# Quick test
# -------------------------

def Test_TinyCNNModel(Steps_in, Steps_out):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    for stpsin, stpout in tqdm(zip(Steps_in, Steps_out)):
        # -------------------------
        # Load training and test data
        # -------------------------
        Xtrain = np.load(f"./Jobs/Data/Xtrain_normalized_{stpsin}_{stpout}.npz")["Data"].astype(np.float32)
        Ytrain = np.load(f"./Jobs/Data/Ytrain_{stpsin}_{stpout}.npz")["y"].astype(np.float32)
        Xtest = np.load(f"./Jobs/Data/Xtest_normalized_{stpsin}_{stpout}.npz")["Data"].astype(np.float32)
        Ytest = np.load(f"./Jobs/Data/Ytest_{stpsin}_{stpout}.npz")["y"].astype(np.float32)

        Ytrain = np.reshape(Ytrain, (len(Ytrain), stpout))
        Ytest = np.reshape(Ytest, (len(Ytest), stpout))

        train_dataset = TensorDataset(torch.tensor(Xtrain), torch.tensor(Ytrain))
        test_dataset = TensorDataset(torch.tensor(Xtest), torch.tensor(Ytest))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # -------------------------
        # Initialize model, loss, optimizer
        # -------------------------
        model = EmbeddedCNN_SOH(stpsin, stpout).to(device)
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
        torch.save(model.state_dict(), f"Models/modelCNNTiny_weights_{stpsin}_{stpout}.pth")
        # -------------------------
        # Cleanup
        # -------------------------
        del model, optimizer, train_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()



Steps_in = [10,20, 25, 25, 25, 25]
Steps_out = [5,10, 25, 50, 100, 200]

Test_TinyCNNModel(Steps_in, Steps_out)
