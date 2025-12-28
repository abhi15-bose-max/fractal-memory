"""
experiments/drift_medium_comparison.py

Compares FractalModel vs ESN vs LSTM vs GRU vs MLP
on MEDIUM-drifting Mackey–Glass data.

Run:
    python -m experiments.drift_medium_comparison
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from src.drift_medium_mg import gen_medium_drift_mg
from src.fractal_model import FractalModel
from src.readout import ridge_regression_fit
from src.reservoir import Reservoir
from src.utils import mse, seed_everything

OUTDIR = "results/plots"
os.makedirs(OUTDIR, exist_ok=True)

SEED = 123
seed_everything(SEED)
DEVICE = "cpu"

T = 3000
TRAIN_RATIO = 0.7
WIN = 50
RES_SIZE = 300
EPOCHS = 20
BATCH = 64
LR = 1e-3
HID = 64


# ----------------------
# Helper: windowing & safe align
# ----------------------
def build_windows(series, win):
    X, Y = [], []
    for i in range(len(series) - win):
        X.append(series[i : i + win])
        Y.append(series[i + win])
    return np.stack(X).astype(np.float32), np.array(Y).astype(np.float32)


def safe_align(preds, series_length, start_idx=0):
    out = np.full(series_length, np.nan, dtype=np.float32)
    if preds is None or len(preds) == 0:
        return out
    start = max(0, start_idx)
    end = min(series_length, start_idx + len(preds))
    src_start = max(0, -start_idx)
    src_end = src_start + (end - start)
    out[start:end] = preds[src_start:src_end]
    return out


# ----------------------
# Generate MEDIUM drift MG
# ----------------------
print("Generating MEDIUM drift MG...")
x = gen_medium_drift_mg(T=T, seed=SEED)

n_train = int(TRAIN_RATIO * T)
x_train = x[:n_train]
x_test = x[n_train:]

X_train, Y_train = build_windows(x_train, WIN)
X_test, Y_test = build_windows(x_test, WIN)


# ----------------------
# 1. FRACTAL MODEL
# ----------------------
print("Running FractalModel...")
fm = FractalModel(
    res_size=RES_SIZE,
    reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": SEED},
    buffer_maxlen=1600,
    online_lr=1e-4,
    device=DEVICE,
)
# consolidation schedule tuned for medium drift
out = fm.run_sequence(x, consolidate_every=300, min_consolidate_samples=50)
preds_slow = out["preds_slow"]
pred_fm = preds_slow[n_train:]  # aligned to x_test
mse_fm = mse(x_test, pred_fm)
print("FractalModel MSE:", mse_fm)


# ----------------------
# 2. ESN + ridge
# ----------------------
print("Running ESN baseline...")
res = Reservoir(in_size=1, res_size=RES_SIZE, spectral_radius=0.95, leak=0.25, seed=SEED)

H = []
for t in range(len(x) - 1):
    h = res.step(float(x[t])).cpu().numpy().copy()
    H.append(h)
H = np.stack(H)
Y_all = x[1:]

H_train = H[: n_train - 1]
Y_train_r = Y_all[: n_train - 1]
H_test = H[n_train - 1 :]
Y_test_r = Y_all[n_train - 1 :]

w_esn, b_esn = ridge_regression_fit(H_train, Y_train_r)
pred_esn = H_test.dot(w_esn) + b_esn
mse_esn = mse(Y_test_r, pred_esn)
print("ESN + ridge MSE:", mse_esn)


# ----------------------
# Window dataset for LSTM/GRU/MLP
# ----------------------
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


# ----------------------
# 3) LSTM baseline
# ----------------------
print("Training LSTM baseline...")


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
        pred = lstm(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if (ep + 1) % 5 == 0:
        print(f"LSTM epoch {ep+1}/{EPOCHS} mean loss {np.mean(losses):.6f}")

lstm.eval()
with torch.no_grad():
    pred_lstm = lstm(torch.tensor(X_test).to(DEVICE)).cpu().numpy()

mse_lstm = mse(Y_test, pred_lstm)
print("LSTM MSE:", mse_lstm)


# ----------------------
# 4) GRU baseline
# ----------------------
print("Training GRU baseline...")


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
        pred = gru(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt_g.step()
        losses.append(loss.item())
    if (ep + 1) % 5 == 0:
        print(f"GRU epoch {ep+1}/{EPOCHS} mean loss {np.mean(losses):.6f}")

gru.eval()
with torch.no_grad():
    pred_gru = gru(torch.tensor(X_test).to(DEVICE)).cpu().numpy()

mse_gru = mse(Y_test, pred_gru)
print("GRU MSE:", mse_gru)


# ----------------------
# 5) MLP baseline
# ----------------------
print("Training MLP baseline...")


class MLPNet(nn.Module):
    def __init__(self, win, hid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(win, hid),
            nn.ReLU(),
            nn.Linear(hid, hid // 2),
            nn.ReLU(),
            nn.Linear(hid // 2, 1),
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
        pred = mlp(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt_m.step()
        losses.append(loss.item())
    if (ep + 1) % 5 == 0:
        print(f"MLP epoch {ep+1}/{EPOCHS} mean loss {np.mean(losses):.6f}")

mlp.eval()
with torch.no_grad():
    pred_mlp = mlp(torch.tensor(X_test).to(DEVICE)).cpu().numpy()

mse_mlp = mse(Y_test, pred_mlp)
print("MLP MSE:", mse_mlp)


# ----------------------
# ALIGN OUTPUTS
# ----------------------
aligned_fm = pred_fm
aligned_esn = safe_align(pred_esn, len(x_test))
aligned_lstm = safe_align(pred_lstm, len(x_test), start_idx=WIN)
aligned_gru = safe_align(pred_gru, len(x_test), start_idx=WIN)
aligned_mlp = safe_align(pred_mlp, len(x_test), start_idx=WIN)


# ----------------------
# PLOTS
# ----------------------
print("Plotting MEDIUM drift results...")

t0 = 50
Tseg = 600
plt.figure(figsize=(12, 4))
plt.plot(x_test[t0 : t0 + Tseg], label="true", color="k", alpha=0.6)
plt.plot(aligned_fm[t0 : t0 + Tseg], label="Fractal", linewidth=1.2)
plt.plot(aligned_esn[t0 : t0 + Tseg], label="ESN", linewidth=1)
plt.plot(aligned_lstm[t0 : t0 + Tseg], label="LSTM", linewidth=1)
plt.plot(aligned_gru[t0 : t0 + Tseg], label="GRU", linewidth=1)
plt.plot(aligned_mlp[t0 : t0 + Tseg], label="MLP", linewidth=1)
plt.legend()
plt.title("MEDIUM Drift MG: Test Segment Predictions")
plt.tight_layout()
p1 = os.path.join(OUTDIR, "medium_drift_segment.png")
plt.savefig(p1, dpi=150)
print("Saved:", p1)
plt.show()

labels = ["Fractal", "ESN", "LSTM", "GRU", "MLP"]
vals = [mse_fm, mse_esn, mse_lstm, mse_gru, mse_mlp]
plt.figure(figsize=(7, 4))
plt.bar(labels, vals)
plt.ylabel("MSE")
plt.title("MEDIUM Drift MG: Model Comparison")
plt.tight_layout()
p2 = os.path.join(OUTDIR, "medium_drift_mse_bar.png")
plt.savefig(p2, dpi=150)
print("Saved:", p2)
plt.show()

# ----------------------
# SUMMARY PRINT
# ----------------------
print("\n=== MEDIUM Drift MSE summary ===")
print("Fractal:", mse_fm)
print("ESN   :", mse_esn)
print("LSTM  :", mse_lstm)
print("GRU   :", mse_gru)
print("MLP   :", mse_mlp)
