"""
experiments/drifting_sine_compare_with_metrics.py

Compare FractalModel, ESN+ridge, LSTM, GRU, MLP
on a drifting sine signal.

Outputs:
- One prediction plot
- Separate bar plots for each metric (NOT normalized)
- Metrics table (txt)

Run:
    python -m experiments.drifting_sine_compare_with_metrics
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from src.fractal_model import FractalModel
from src.reservoir import Reservoir
from src.readout import ridge_regression_fit
from src.utils import seed_everything

# drifting sine generator (must exist)
from src.data import gen_drifting_sine

# -----------------------
# Config
# -----------------------
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

OUTDIR = "results/plots/drifting_sine"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------
# Metrics
# -----------------------
def mse(y, p): return float(np.mean((y - p) ** 2))
def mae(y, p): return float(np.mean(np.abs(y - p)))
def smape(y, p, eps=1e-8):
    return float(np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + eps)))

def dtw_distance(a, b):
    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf)
    D[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = (a[i-1] - b[j-1])**2
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return float(D[n, m])

def estimate_lag(y, p, max_lag=200):
    yt = y - y.mean()
    yp = p - p.mean()
    corr = np.correlate(yt, yp, mode="full")
    mid = len(corr) // 2
    lo, hi = mid - max_lag, mid + max_lag + 1
    return int(np.argmax(corr[lo:hi]) + lo - mid)

# -----------------------
# Models
# -----------------------
class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(1, HID, batch_first=True)
        self.fc = nn.Linear(HID, 1)
    def forward(self, x):
        o, _ = self.rnn(x.unsqueeze(-1))
        return self.fc(o[:, -1]).squeeze()

class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(1, HID, batch_first=True)
        self.fc = nn.Linear(HID, 1)
    def forward(self, x):
        o, _ = self.rnn(x.unsqueeze(-1))
        return self.fc(o[:, -1]).squeeze()

class MLPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WIN, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze()

def train(model, X, Y):
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        opt.step()

# -----------------------
# Data
# -----------------------
print("Generating drifting sine...")
x = gen_drifting_sine(T=T, seed=SEED)
split = int(TRAIN_RATIO * T)
x_train, x_test = x[:split], x[split:]

def make_windows(x):
    X, Y = [], []
    for i in range(len(x) - WIN):
        X.append(x[i:i+WIN])
        Y.append(x[i+WIN])
    return np.array(X), np.array(Y)

Xtr, Ytr = make_windows(x_train)
Xte, Yte = make_windows(x_test)

Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
Xte_t = torch.tensor(Xte, dtype=torch.float32)

# -----------------------
# Fractal Memory
# -----------------------
fm = FractalModel(
    res_size=RES_SIZE,
    reservoir_kwargs={"spectral_radius":0.95, "leak":0.25, "seed":SEED},
    buffer_maxlen=1500,
    online_lr=1e-4,
)
out = fm.run_sequence(x, consolidate_every=300)
pred_fm = out["preds_slow"][split:]

# -----------------------
# ESN
# -----------------------
res = Reservoir(1, RES_SIZE, spectral_radius=0.95, leak=0.25, seed=SEED)
H = [res.step(x[t]).cpu().numpy() for t in range(len(x)-1)]
H = np.stack(H)
w, b = ridge_regression_fit(H[:split-1], x[1:split])
pred_esn = H[split-1:] @ w + b

# -----------------------
# Neural baselines
# -----------------------
lstm = LSTMNet()
gru = GRUNet()
mlp = MLPNet()

train(lstm, Xtr_t, Ytr_t)
train(gru, Xtr_t, Ytr_t)
train(mlp, Xtr_t, Ytr_t)

pred_lstm = lstm(Xte_t).detach().numpy()
pred_gru  = gru(Xte_t).detach().numpy()
pred_mlp  = mlp(Xte_t).detach().numpy()

# -----------------------
# Align predictions
# -----------------------
L = min(len(pred_fm), len(pred_esn), len(pred_lstm))
y_true = x_test[:L]

preds = {
    "Fractal": pred_fm[:L],
    "ESN": pred_esn[:L],
    "LSTM": pred_lstm[:L],
    "GRU": pred_gru[:L],
    "MLP": pred_mlp[:L],
}

# -----------------------
# Metrics
# -----------------------
metrics = {}
for k, p in preds.items():
    metrics[k] = {
        "MSE": mse(y_true, p),
        "MAE": mae(y_true, p),
        "sMAPE": smape(y_true, p),
        "DTW": dtw_distance(y_true[:800], p[:800]),
        "Lag": estimate_lag(y_true, p),
    }

# -----------------------
# Plots (SEPARATE, RAW)
# -----------------------
for metric in ["MSE", "MAE", "sMAPE", "DTW", "Lag"]:
    plt.figure(figsize=(6,4))
    plt.bar(metrics.keys(), [metrics[m][metric] for m in metrics])
    plt.ylabel(metric)
    plt.title(f"Drifting Sine – {metric}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"drifting_sine_{metric.lower()}.png"), dpi=200)
    plt.close()

# -----------------------
# Save metrics table
# -----------------------
with open(os.path.join(OUTDIR, "metrics.txt"), "w") as f:
    for m, d in metrics.items():
        f.write(f"{m}:\n")
        for k, v in d.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\n")

print("Done. Results saved to:", OUTDIR)

# -----------------------
# Console summary
# -----------------------
print("\n=== Drifting Sine: Metric Summary ===")
for model, d in metrics.items():
    print(f"\n{model}")
    print(f"  MSE   : {d['MSE']:.6e}")
    print(f"  MAE   : {d['MAE']:.6e}")
    print(f"  sMAPE : {d['sMAPE']:.6e}")
    print(f"  DTW   : {d['DTW']:.6e}")
    print(f"  Lag   : {d['Lag']}")

