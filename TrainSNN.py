import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools
import struct
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import time
import gc
from tensorflow.keras.models import load_model
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
DataPath = "./Data/"
Train_data = [
    "cell1-25.h5",
    "cell2-35.h5",
    "cell3-45.h5",
]
Test_data = [
    "cell7-25.h5",
    "cell8-35.h5",
    "cell9-45.h5",
    "cell13-35.h5",
    "cell14-45.h5",
    "cell15-45.h5",
    "cell16-45.h5",
    "cell17-45.h5",
    "cell18-45.h5",
    "cell19-45.h5",
]
Fast_data = [
    "cell1-25.h5",
    "cell2-35.h5",
    "cell3-45.h5",
    "cell7-25.h5",
    "cell8-35.h5",
    "cell9-45.h5",
    "cell13-35.h5",
    "cell14-45.h5",
    "cell15-45.h5",
    "cell16-45.h5",
    "cell17-45.h5",
    "cell18-45.h5",
    "cell19-45.h5",
]
Normal_data = [
    "cell10-25.h5",
    "cell11-35.h5",
    "cell12-45.h5",
    "cell4-25.h5",
    "cell5-35.h5",
    "cell6-45.h5",
]

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available. PyTorch is running on CPU.")

dtype = torch.float
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


maxI, minI, maxV, minV, maxT, minT = [
    9.381874,
    -20.00478,
    3.600254,
    1.989369,
    57.09709,
    18.68196,
]


def Sequence_Split_SOH(sequence, X, y, step_in, step_out):
    for i in range(len(sequence)):
        end_ix = i + step_in
        out_end_ix = end_ix + step_out

        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(
            seq_x.drop(
                [
                    "SOC",
                    "SOH",
                    "Qd",
                    "QC",
                    "Time[h]",
                    "I[A]",
                    "U[V]",
                    "Cycle_index",
                    "T1[°C]",
                ],
                axis=1,
            ).to_numpy()
        )
        y.append(seq_y["SOH"].to_numpy())
    return X, y


def NormalizationGPU(data, stepin, maxlength=2000, device="cuda"):
    nd = []
    for i in range(len(data)):
        im = []
        vm = []
        tm = []
        for c in range(len(data[i])):
            norm_i = (data[i][c][0] - minI) / (maxI - minI)
            norm_v = (data[i][c][1] - minV) / (maxV - minV)
            norm_t = (data[i][c][2] - minT) / (maxT - minT)

            t_i = torch.zeros(maxlength, device=device)
            t_i[:min(maxlength, len(norm_i))] = torch.tensor(
                norm_i[:min(maxlength, len(norm_i))], device=device
            )
            im.append(t_i)

            t_v = torch.zeros(maxlength, device=device)
            t_v[:min(maxlength, len(norm_v))] = torch.tensor(
                norm_v[:min(maxlength, len(norm_v))], device=device
            )
            vm.append(t_v)

            t_t = torch.zeros(maxlength, device=device)
            t_t[:min(maxlength, len(norm_t))] = torch.tensor(
                norm_t[:min(maxlength, len(norm_t))], device=device
            )
            tm.append(t_t)

        d = torch.stack([torch.stack(im), torch.stack(vm), torch.stack(tm)], dim=0)
        d = d.view(3, stepin * maxlength).T.reshape(-1, 3)
        nd.append(d)
    return nd

def EncodingGPU(data, stpsin, maxlength=2000, device="cuda"):
    nd = []
    r = stpsin * maxlength - 1
    for i in range(len(data)):
        current = data[i].to(device)
        dffi = torch.abs(torch.diff(current[:, 0]))
        dffv = torch.abs(torch.diff(current[:, 1]))
        dfft = torch.abs(torch.diff(current[:, 2]))

        im_seq = (dffi >= 0.0002).float()
        vm_seq = (dffv >= 0.0002).float()
        tm_seq = (dfft >= 0.002).float()

        d = torch.stack([im_seq, vm_seq, tm_seq], dim=0).T.reshape(r, 3)
        nd.append(d)
    return nd


loss2 = nn.MSELoss()
loss = nn.L1Loss()


def measure_accuracy(model, dataloader):
    t = []
    errors = []
    with torch.no_grad():
        model.to(device)
        model.eval()
        for data, targets in iter(dataloader):
            data = data.to(device)
            targets = targets.to(device)

            # forward-pass
            t1 = time.time()
            spk_rec, _ = model(data)
            t2 = time.time()
            t.append(t2 - t1)

            # compute errors for all batches
            batch_errors = torch.abs(spk_rec[0] - targets)
            errors.append(batch_errors.cpu().numpy())

            # loss (optional, still keep)
            loss_mae = torch.mean(batch_errors)
            loss_mse = torch.mean((spk_rec[0] - targets) ** 2)
            loss_rmse = torch.sqrt(loss_mse)

    errors = np.concatenate(errors).flatten()
    mae = np.mean(errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    std = np.std(errors)
    ci95 = 1.96 * std / np.sqrt(len(errors))
    avg_time = np.mean(t)

    return mae, mse, rmse, std, ci95, avg_time


from snntorch import surrogate


class Net(nn.Module):
    def __init__(self, num_hidden, num_outputs, num_inputs):
        super().__init__()
        beta1 = torch.rand(num_hidden)
        beta2 = torch.rand(num_outputs)
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(
            beta=beta1, learn_beta=True, learn_threshold=True
        )  # spike_grad=sigmoid, reset_mechanism,,learn_beta=True
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(
            beta=beta2, learn_beta=True, learn_threshold=True
        )  # learn_beta=True, threshold=1

        # Add a final layer to ensure the output values are between 0 and 1
        self.fc3 = nn.Linear(
            num_outputs, num_outputs
        )  # To map the output to the final space
        self.sigmoid = (
            nn.Sigmoid()
        )  # Sigmoid activation to bound the values between 0 and 1

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        # time-loop
        for step in range(1):
            cur1 = self.fc1(x.flatten(1))  # batch32 x (3 x 37500)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Apply final layer and sigmoid activation to ensure output is between 0 and 1
            cur3 = self.fc3(spk2)
            spk2_sigmoid = self.sigmoid(
                cur3
            )  # Ensure output values are between 0 and 1

            # store in list
            spk2_rec.append(spk2_sigmoid)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(
            mem2_rec, dim=0
        )  # time-steps x batch x num_out


class GPUDataset(torch.utils.data.Dataset):
    """
    Dataset that keeps sequences on CPU and transfers each batch to GPU on the fly.
    """
    def __init__(self, X_list, Y_tensor):
        self.X_list = X_list  # list of CPU tensors
        self.Y_tensor = Y_tensor  # CPU tensor

    def __len__(self):
        return len(self.Y_tensor)

    def __getitem__(self, idx):
        X = self.X_list[idx].float().to(device)  # transfer only one sample to GPU
        Y = self.Y_tensor[idx].float().to(device)
        return X, Y


def Test_ModelGPU(Steps_in, Steps_out, maxlength=1500, batch_size=32):
    for stpsin, stpout in tqdm(zip(Steps_in, Steps_out)):
        # -------------------------
        # Load & prepare training data (CPU)
        # -------------------------
        Xtrain, YTrain = [], []
        for n in Train_data:
            df = pd.read_hdf(DataPath + n[0:-3] + "_charge.h5", key="df")
            df["SOH"] = savgol_filter(df["SOH"], window_length=500, polyorder=2)
            Sequence_Split_SOH(df, Xtrain, YTrain, stpsin, stpout)

        Xtest, YTest = [], []
        for n in Test_data:
            df = pd.read_hdf(DataPath + n[0:-3] + "_charge.h5", key="df")
            df["SOH"] = savgol_filter(df["SOH"], window_length=500, polyorder=2)
            Sequence_Split_SOH(df, Xtest, YTest, stpsin, stpout)

        # -------------------------
        # Normalize & encode on GPU
        # -------------------------
        Xtrain_norm = NormalizationGPU(Xtrain, stpsin, maxlength=maxlength, device=device)
        XtrainSNN = EncodingGPU(Xtrain_norm, stpsin, maxlength=maxlength, device=device)
        YTrain = torch.tensor(np.reshape(YTrain, (len(YTrain), stpout)), dtype=torch.float32, device=device)

        Xtest_norm = NormalizationGPU(Xtest, stpsin, maxlength=maxlength, device=device)
        XtestSNN = EncodingGPU(Xtest_norm, stpsin, maxlength=maxlength, device=device)
        YTest = torch.tensor(np.reshape(YTest, (len(YTest), stpout)), dtype=torch.float32, device=device)
        Xtrain_norm_np = [x.detach().cpu().numpy() for x in Xtrain_norm]
        XtrainSNN_np   = [x.detach().cpu().numpy() for x in XtrainSNN]
        Xtest_norm_np  = [x.detach().cpu().numpy() for x in Xtest_norm]
        XtestSNN_np    = [x.detach().cpu().numpy() for x in XtestSNN]

        # Move YTrain/YTest to CPU NumPy
        YTrain_np = YTrain.detach().cpu().numpy()
        YTest_np  = YTest.detach().cpu().numpy()

        # Save normalized and encoded data
        np.savez_compressed(f"./Data/Xtrain_normalized_{stpsin}_{stpout}.npz", Data=Xtrain_norm_np)
        np.savez_compressed(f"./Data/Ytrain_{stpsin}_{stpout}.npz", y=YTrain_np)
        np.savez_compressed(f"./Data/XtrainSNN_{stpsin}_{stpout}.npz", Data=XtrainSNN_np)

        np.savez_compressed(f"./Data/Xtest_normalized_{stpsin}_{stpout}.npz", Data=Xtest_norm_np)
        np.savez_compressed(f"./Data/Ytest_{stpsin}_{stpout}.npz", y=YTest_np)
        np.savez_compressed(f"./Data/XtestSNN_{stpsin}_{stpout}.npz", Data=XtestSNN_np)        

        # -------------------------
        # Convert to CPU lists for batch-wise loading to avoid OOM
        # -------------------------
        Xtrain_list = [x.cpu() for x in XtrainSNN]
        Xtest_list = [x.cpu() for x in XtestSNN]

        # -------------------------
        # Prepare DataLoaders
        # -------------------------
        train_dataset = GPUDataset(Xtrain_list, YTrain.cpu())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = GPUDataset(Xtest_list, YTest.cpu())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # -------------------------
        # Build SNN model
        # -------------------------
        num_inputs = 3 * (stpsin * maxlength - 1)
        num_hidden = 1000
        num_outputs = stpout
        net = Net(num_hidden, num_outputs, num_inputs).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
        loss_fn = nn.L1Loss()

        # -------------------------
        # Training loop
        # -------------------------
        num_epochs = 50
        for epoch in range(num_epochs):
            net.train()
            for data, targets in train_loader:
                spk_rec, _ = net(data)
                loss_val = loss_fn(spk_rec[0], targets)
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Save model
        torch.save(net.state_dict(), f"Models/modelSNN_weights_{stpsin}_{stpout}.pth")

        
        # -------------------------
        # Cleanup
        # -------------------------
        del net, optimizer, train_loader, test_loader
        del Xtrain_list, YTrain, Xtest_list, YTest
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

Steps_in = [1,10,20, 25, 25, 25, 25]
Steps_out = [1,5,10, 25, 50, 100, 200]

Test_ModelGPU(Steps_in, Steps_out, maxlength=1500)
