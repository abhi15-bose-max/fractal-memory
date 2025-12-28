"""
experiments/mg_baseline_comparison.py

Compare FractalModel vs simple baselines (ESN/ridge, LSTM, GRU, MLP) on Mackey-Glass.
Produces:
- MSE metrics
- aligned prediction plots
- MSE bar chart

Run:
    python -m experiments.mg_baseline_comparison
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from src.mackey_glass import gen_mackey_glass_sequence
from src.fractal_model import FractalModel
from src.readout import ridge_regression_fit
from src.reservoir import Reservoir
from src.utils import mse, seed_everything


# -----------------------
# Config
# -----------------------
OUTDIR = "results/plots"
os.makedirs(OUTDIR, exist_ok=True)

SEED = 123
seed_everything(SEED)
DEVICE = "cpu"

T = 3000
TRAIN_RATIO = 0.7
WIN = 50  # sliding window
RES_SIZE = 300

# LSTM/GRU/MLP params
HID = 64
EPOCHS = 30
BATCH = 64
LR = 1e-3


# -----------------------
# Data
# -----------------------
print("Generating Mackey-Glass data...")
x = gen_mackey_glass_sequence(T=T, seed=SEED).astype(np.float32)

n_train = int(TRAIN_RATIO * T)
x_train = x[:n_train]
x_test = x[n_train:]


def build_windows(series, win):
    X, Y = [], []
    for i in range(len(series) - win):
        X.append(series[i:i + win])
        Y.append(series[i + win])
    return np.stack(X).astype(np.float32), np.array(Y).astype(np.float32)


X_train, Y_train = build_windows(x_train, WIN)
X_test, Y_test = build_windows(x_test, WIN)


# -----------------------
# Safe alignment helper
# -----------------------
def safe_align(preds, series_length, start_idx=0):
    """
    preds: prediction array
    series_length: length of target series (e.g., len(x_test))
    start_idx: index where preds[0] should align

    Returns array of length series_length, filling unreachable
    indices with NaN, clipping automatically to prevent slicing errors.
    """
    out = np.full(series_length, np.nan, dtype=np.float32)
    if preds is None or len(preds) == 0:
        return out

    # compute safe slice boundaries
    start = max(0, start_idx)
    end = min(series_length, start_idx + len(preds))

    # compute source indices (skip if start_idx negative)
    src_start = max(0, -start_idx)
    src_end = src_start + (end - start)

    out[start:end] = preds[src_start:src_end]
    return out


# -----------------------
# 1) FractalModel (full)
# -----------------------
print("Running FractalModel (full) ...")
fm = FractalModel(
    res_size=RES_SIZE,
    reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": SEED},
    buffer_maxlen=1200,
    online_lr=1e-4,
    device=DEVICE,
)
out = fm.run_sequence(x, consolidate_every=350, min_consolidate_samples=60)
preds_slow = out["preds_slow"]
pred_fm = preds_slow[n_train:]  # aligns with x_test

mse_fm = mse(x_test, pred_fm)
print(f"FractalModel (slow) MSE on test portion: {mse_fm:.6f}")


# -----------------------
# 2) ESN baseline (reservoir-only + ridge)
# -----------------------
print("Running ESN baseline...")
res = Reservoir(
    in_size=1, res_size=RES_SIZE,
    spectral_radius=0.95, leak=0.25, seed=SEED
)

H_all = []
for t in range(len(x) - 1):
    h = res.step(float(x[t])).cpu().numpy().copy()
    H_all.append(h)

H_all = np.stack(H_all)
Y_all = x[1:]

H_train = H_all[:n_train - 1]
Y_train_r = Y_all[:n_train - 1]

H_test = H_all[n_train - 1:]
Y_test_r = Y_all[n_train - 1:]

w_esn, b_esn = ridge_regression_fit(H_train, Y_train_r, alpha=1e-3)
pred_esn = H_test.dot(w_esn) + b_esn

mse_esn = mse(Y_test_r, pred_esn)
print(f"ESN + ridge MSE: {mse_esn:.6f}")


# -----------------------
# Dataset for PyTorch models
# -----------------------
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


train_ds = WindowDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True)
loss_fn = nn.MSELoss()


# -----------------------
# 3) LSTM baseline
# -----------------------
print("Training LSTM...")
class LSTMNet(nn.Module):
    def __init__(self, win, hid):
        super().__init__()
        self.rnn = nn.LSTM(1, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        h = out[:, -1, :]
        return self.fc(h).squeeze(-1)


lstm = LSTMNet(WIN, HID).to(DEVICE)
opt = optim.Adam(lstm.parameters(), lr=LR)

for ep in range(EPOCHS):
    lstm.train()
    losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        outp = lstm(xb)
        loss = loss_fn(outp, yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if (ep + 1) % 10 == 0:
        print(f"LSTM epoch {ep+1}/{EPOCHS} mean loss {np.mean(losses):.6f}")

lstm.eval()
with torch.no_grad():
    pred_lstm = lstm(torch.tensor(X_test).to(DEVICE)).cpu().numpy()

mse_lstm = mse(Y_test, pred_lstm)
print(f"LSTM MSE: {mse_lstm:.6f}")


# -----------------------
# 4) GRU baseline
# -----------------------
print("Training GRU...")

class GRUNet(nn.Module):
    def __init__(self, win, hid):
        super().__init__()
        self.rnn = nn.GRU(1, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        h = out[:, -1, :]
        return self.fc(h).squeeze(-1)


gru = GRUNet(WIN, HID).to(DEVICE)
opt_g = optim.Adam(gru.parameters(), lr=LR)

for ep in range(EPOCHS):
    gru.train()
    losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_g.zero_grad()
        outp = gru(xb)
        loss = loss_fn(outp, yb)
        loss.backward()
        opt_g.step()
        losses.append(loss.item())
    if (ep + 1) % 10 == 0:
        print(f"GRU epoch {ep+1}/{EPOCHS} mean loss {np.mean(losses):.6f}")

gru.eval()
with torch.no_grad():
    pred_gru = gru(torch.tensor(X_test).to(DEVICE)).cpu().numpy()

mse_gru = mse(Y_test, pred_gru)
print(f"GRU MSE: {mse_gru:.6f}")


# -----------------------
# 5) MLP baseline
# -----------------------
print("Training MLP...")
class MLPNet(nn.Module):
    def __init__(self, win, hid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(win, hid),
            nn.ReLU(),
            nn.Linear(hid, hid // 2),
            nn.ReLU(),
            nn.Linear(hid // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


mlp = MLPNet(WIN, HID).to(DEVICE)
opt_m = optim.Adam(mlp.parameters(), lr=LR)

for ep in range(EPOCHS):
    mlp.train()
    losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_m.zero_grad()
        outp = mlp(xb)
        loss = loss_fn(outp, yb)
        loss.backward()
        opt_m.step()
        losses.append(loss.item())
    if (ep + 1) % 10 == 0:
        print(f"MLP epoch {ep+1}/{EPOCHS} mean loss {np.mean(losses):.6f}")

mlp.eval()
with torch.no_grad():
    pred_mlp = mlp(torch.tensor(X_test).to(DEVICE)).cpu().numpy()

mse_mlp = mse(Y_test, pred_mlp)
print(f"MLP MSE: {mse_mlp:.6f}")


# -----------------------
# Align predictions safely
# -----------------------
aligned_fm = pred_fm                       # already correct length
aligned_esn = safe_align(pred_esn, len(x_test), start_idx=0)
aligned_lstm = safe_align(pred_lstm, len(x_test), start_idx=WIN)
aligned_gru  = safe_align(pred_gru,  len(x_test), start_idx=WIN)
aligned_mlp  = safe_align(pred_mlp,  len(x_test), start_idx=WIN)


# -----------------------
# Print summary
# -----------------------
print("\n=== Test MSE summary ===")
print(f"FractalModel (slow): {mse_fm:.6f}")
print(f"ESN + ridge        : {mse_esn:.6f}")
print(f"LSTM               : {mse_lstm:.6f}")
print(f"GRU                : {mse_gru:.6f}")
print(f"MLP                : {mse_mlp:.6f}")


# -----------------------
# Plot: test segment predictions
# -----------------------
t0 = 50
Tplot = 600

plt.figure(figsize=(12, 4))
plt.plot(x_test[t0:t0 + Tplot], label="true", color="k", alpha=0.6)
plt.plot(aligned_fm[t0:t0 + Tplot], label="Fractal", linewidth=1)
plt.plot(aligned_esn[t0:t0 + Tplot], label="ESN", linewidth=1)
plt.plot(aligned_lstm[t0:t0 + Tplot], label="LSTM", linewidth=1)
plt.plot(aligned_gru[t0:t0 + Tplot], label="GRU", linewidth=1)
plt.plot(aligned_mlp[t0:t0 + Tplot], label="MLP", linewidth=1)
plt.legend()
plt.title("Mackey-Glass: Test Segment Predictions")
plt.tight_layout()

p1 = os.path.join(OUTDIR, "mg_baseline_segment.png")
plt.savefig(p1, dpi=150)
print("Saved:", p1)
plt.show()


# -----------------------
# Plot: MSE bar chart
# -----------------------
labels = ["Fractal", "ESN", "LSTM", "GRU", "MLP"]
vals = [mse_fm, mse_esn, mse_lstm, mse_gru, mse_mlp]

plt.figure(figsize=(7, 4))
plt.bar(labels, vals)
plt.ylabel("MSE")
plt.title("Mackey-Glass Baseline Comparison")
plt.tight_layout()

p2 = os.path.join(OUTDIR, "mg_baseline_mse_bar.png")
plt.savefig(p2, dpi=150)
print("Saved:", p2)
plt.show()
