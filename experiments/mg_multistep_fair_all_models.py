"""
Fair multi-step forecasting on Mackey–Glass
k = 1..30 (recursive rollout)

Models:
- Fractal Memory
- ESN (reservoir + ridge)
- LSTM
- GRU
- MLP

Metric:
- RMSE vs forecast horizon

Run:
    python -m experiments.mg_multistep_fair_all_models
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.mackey_glass import gen_mackey_glass_sequence
from src.reservoir import Reservoir
from src.fractal_model import FractalModel
from src.readout import ridge_regression_fit

# -----------------------
# Config
# -----------------------
T = 3000
TRAIN_FRAC = 0.7
K_MAX = 30
RES_SIZE = 300
WINDOW = 20
HIDDEN = 64
EPOCHS = 25
DEVICE = "cpu"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

SAVE_DIR = "results/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------
# Utility
# -----------------------
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

# -----------------------
# Data
# -----------------------
x = gen_mackey_glass_sequence(T=T, seed=SEED)
split = int(TRAIN_FRAC * T)
x_train, x_test = x[:split], x[split:]

# ============================================================
# ESN (1-step trained, recursive rollout)
# ============================================================
res_esn = Reservoir(in_size=1, res_size=RES_SIZE, seed=SEED)

H, Y = [], []
for t in range(len(x_train) - 1):
    h = res_esn.step(x_train[t]).cpu().numpy()
    H.append(h)
    Y.append(x_train[t + 1])

H = np.stack(H)
Y = np.array(Y)
w_esn, b_esn = ridge_regression_fit(H, Y)

def esn_step(x_t):
    h = res_esn.step(x_t).cpu().numpy()
    return h @ w_esn + b_esn

# ============================================================
# Fractal Memory (1-step, recursive)
# ============================================================
fm = FractalModel(
    res_size=RES_SIZE,
    buffer_maxlen=1200,
    online_lr=1e-4
)
fm.run_sequence(x_train, consolidate_every=300)

def fractal_step(x_t):
    return fm.step(x_t)[0]

# ============================================================
# Windowed datasets for NN baselines
# ============================================================
def make_windows(series, win):
    X, Y = [], []
    for i in range(len(series) - win):
        X.append(series[i:i+win])
        Y.append(series[i+win])
    return np.array(X), np.array(Y)

Xtr, Ytr = make_windows(x_train, WINDOW)
Xte, Yte = make_windows(x_test, WINDOW)

Xtr_t = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(-1)
Ytr_t = torch.tensor(Ytr, dtype=torch.float32)

# ============================================================
# Models
# ============================================================
class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(1, HIDDEN, batch_first=True)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1]).squeeze()

class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(1, HIDDEN, batch_first=True)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1]).squeeze()

class MLPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WINDOW, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x.squeeze(-1)).squeeze()

def train(model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), Ytr_t)
        loss.backward()
        opt.step()

lstm = LSTMNet().to(DEVICE)
gru = GRUNet().to(DEVICE)
mlp = MLPNet().to(DEVICE)

train(lstm)
train(gru)
train(mlp)

# ============================================================
# Recursive multi-step evaluation (FAIR)
# ============================================================
ks = np.arange(1, K_MAX + 1)

rmse_esn, rmse_fm, rmse_lstm, rmse_gru, rmse_mlp = [], [], [], [], []

for k in ks:
    pe, pf, pl, pg, pm, tgt = [], [], [], [], [], []

    for t in range(len(x_test) - WINDOW - k):
        # ground truth
        tgt.append(x_test[t + WINDOW + k])

        # ESN
        x_hat = x_test[t]
        for _ in range(k):
            x_hat = esn_step(x_hat)
        pe.append(x_hat)

        # Fractal
        x_hat = x_test[t]
        for _ in range(k):
            x_hat = fractal_step(x_hat)
        pf.append(x_hat)

        # NN baselines
        xin = x_test[t:t+WINDOW].copy()
        for _ in range(k):
            y = lstm(torch.tensor(xin).view(1, WINDOW, 1)).item()
            xin = np.roll(xin, -1)
            xin[-1] = y
        pl.append(y)

        xin = x_test[t:t+WINDOW].copy()
        for _ in range(k):
            y = gru(torch.tensor(xin).view(1, WINDOW, 1)).item()
            xin = np.roll(xin, -1)
            xin[-1] = y
        pg.append(y)

        xin = x_test[t:t+WINDOW].copy()
        for _ in range(k):
            y = mlp(torch.tensor(xin).view(1, WINDOW, 1)).item()
            xin = np.roll(xin, -1)
            xin[-1] = y
        pm.append(y)

    tgt = np.array(tgt)

    rmse_esn.append(rmse(pe, tgt))
    rmse_fm.append(rmse(pf, tgt))
    rmse_lstm.append(rmse(pl, tgt))
    rmse_gru.append(rmse(pg, tgt))
    rmse_mlp.append(rmse(pm, tgt))

# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(9, 5))
plt.plot(ks, rmse_fm, marker="o", label="Fractal Memory")
plt.plot(ks, rmse_esn, marker="s", label="ESN")
plt.plot(ks, rmse_lstm, marker="^", label="LSTM")
plt.plot(ks, rmse_gru, marker="d", label="GRU")
plt.plot(ks, rmse_mlp, marker="x", label="MLP")

plt.xlabel("Forecast horizon k")
plt.ylabel("RMSE")
plt.title("Mackey–Glass Multi-step Forecasting (Fair, Recursive)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

outpath = os.path.join(SAVE_DIR, "mg_multistep_fair_all_models.png")
plt.savefig(outpath, dpi=200)
plt.show()

print("Saved:", outpath)

print("\nRMSE @ k = 1")
print("Fractal:", rmse_fm[0])
print("ESN    :", rmse_esn[0])
print("LSTM   :", rmse_lstm[0])
print("GRU    :", rmse_gru[0])
print("MLP    :", rmse_mlp[0])
