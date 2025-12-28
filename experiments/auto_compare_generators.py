"""
experiments/auto_compare_generators.py

Automatically run baseline comparisons (FractalModel, ESN+ridge, LSTM, GRU, MLP)
across multiple signal generators placed under src/.

Run:
    python -m experiments.auto_compare_generators
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Import your models/utilities
from src.fractal_model import FractalModel
from src.readout import ridge_regression_fit
from src.reservoir import Reservoir
from src.utils import mse, seed_everything

# Import generators (some may be optional)
from src.drift_fast_mg import gen_fast_drift_mg
from src.drift_medium_mg import gen_medium_drift_mg
from src.drift_slow_mg import gen_slow_drift_mg
from src.drift_multiscale import gen_multiscale_drift
from src.drift_piecewise import gen_piecewise
from src.drift_randomwalk import gen_randomwalk_freq_sine
from src.drift_am_fm import gen_am_fm
from src.drift_heteroskedastic import gen_heteroskedastic_signal
from src.drift_composite import gen_composite_signal

# optional: if you have standard mackey glass
try:
    from src.mackey_glass import gen_mackey_glass_sequence
    HAS_MG = True
except Exception:
    HAS_MG = False

# -----------------------
# Config
# -----------------------
OUTROOT = "results/plots"
os.makedirs(OUTROOT, exist_ok=True)

SEED = 123
seed_everything(SEED)
DEVICE = "cpu"

# Common dataset / model params
T = 3000
TRAIN_RATIO = 0.7
WIN = 50
RES_SIZE = 300

EPOCHS = 20
BATCH = 64
LR = 1e-3
HID = 64

# List of (label, callable, kwargs)
GENERATORS = [
    ("fast_mg", gen_fast_drift_mg, {"T": T, "seed": SEED}),
    ("medium_mg", gen_medium_drift_mg, {"T": T, "seed": SEED}),
    ("slow_mg", gen_slow_drift_mg, {"T": T, "seed": SEED}),
    ("multiscale_sine", gen_multiscale_drift, {"T": T, "seed": SEED}),
    ("piecewise", gen_piecewise, {"T": T, "seed": SEED}),
    ("randomwalk_sine", gen_randomwalk_freq_sine, {"T": T, "seed": SEED}),
    ("am_fm", gen_am_fm, {"T": T, "seed": SEED}),
    ("heteroskedastic", gen_heteroskedastic_signal, {"T": T, "seed": SEED}),
    ("composite", gen_composite_signal, {"T": T, "seed": SEED}),
]

if HAS_MG:
    GENERATORS.insert(0, ("mackey_glass", gen_mackey_glass_sequence, {"T": T, "seed": SEED}))


# -----------------------
# Helpers
# -----------------------
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


# Lightweight model classes reused
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


# Training helper for window-based nets
def train_window_model(model, train_loader, epochs=EPOCHS, lr=LR, device="cpu"):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return model


# Run one generator experiment
def run_experiment(gen_label, gen_fn, gen_kwargs):
    print(f"\n--- Running experiment: {gen_label} ---")
    x = gen_fn(**gen_kwargs)
    n_train = int(TRAIN_RATIO * len(x))
    x_train = x[:n_train]
    x_test = x[n_train:]

    # Prepare windows
    X_train, Y_train = build_windows(x_train, WIN)
    X_test, Y_test = build_windows(x_test, WIN)

    # 1) FractalModel
    fm = FractalModel(
        res_size=RES_SIZE,
        reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": SEED},
        buffer_maxlen=1400,
        online_lr=1e-4,
        device=DEVICE,
    )
    out = fm.run_sequence(x, consolidate_every=300, min_consolidate_samples=50)
    preds_slow = out["preds_slow"]
    pred_fm = preds_slow[n_train:]
    mse_fm = mse(x_test, pred_fm)
    print("FractalModel MSE:", mse_fm)

    # 2) ESN + ridge
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

    # PyTorch window dataset
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

    # LSTM
    lstm = LSTMNet(WIN, HID)
    lstm = train_window_model(lstm, train_loader, epochs=EPOCHS, lr=LR, device=DEVICE)
    lstm.eval()
    with torch.no_grad():
        pred_lstm = lstm(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
    mse_lstm = mse(Y_test, pred_lstm)
    print("LSTM MSE:", mse_lstm)

    # GRU
    gru = GRUNet(WIN, HID)
    gru = train_window_model(gru, train_loader, epochs=EPOCHS, lr=LR, device=DEVICE)
    gru.eval()
    with torch.no_grad():
        pred_gru = gru(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
    mse_gru = mse(Y_test, pred_gru)
    print("GRU MSE:", mse_gru)

    # MLP
    mlp = MLPNet(WIN, HID)
    mlp = train_window_model(mlp, train_loader, epochs=EPOCHS, lr=LR, device=DEVICE)
    mlp.eval()
    with torch.no_grad():
        pred_mlp = mlp(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
    mse_mlp = mse(Y_test, pred_mlp)
    print("MLP MSE:", mse_mlp)

    # Align predictions (FractalModel pred_fm already aligned)
    aligned_fm = pred_fm
    aligned_esn = safe_align(pred_esn, len(x_test), start_idx=0)
    aligned_lstm = safe_align(pred_lstm, len(x_test), start_idx=WIN)
    aligned_gru = safe_align(pred_gru, len(x_test), start_idx=WIN)
    aligned_mlp = safe_align(pred_mlp, len(x_test), start_idx=WIN)

    # Plots directory for this generator
    outdir = os.path.join(OUTROOT, gen_label)
    os.makedirs(outdir, exist_ok=True)

    # Plot a test segment
    t0 = 50
    Tplot = min(800, len(x_test) - t0)
    plt.figure(figsize=(12, 4))
    plt.plot(x_test[t0 : t0 + Tplot], label="true", color="k", alpha=0.6)
    plt.plot(aligned_fm[t0 : t0 + Tplot], label="Fractal", linewidth=1.2)
    plt.plot(aligned_esn[t0 : t0 + Tplot], label="ESN", linewidth=1)
    plt.plot(aligned_lstm[t0 : t0 + Tplot], label="LSTM", linewidth=1)
    plt.plot(aligned_gru[t0 : t0 + Tplot], label="GRU", linewidth=1)
    plt.plot(aligned_mlp[t0 : t0 + Tplot], label="MLP", linewidth=1)
    plt.legend()
    plt.title(f"{gen_label}: Test Segment Predictions")
    plt.tight_layout()
    p1 = os.path.join(outdir, f"{gen_label}_segment.png")
    plt.savefig(p1, dpi=150)
    plt.close()

    # MSE bar chart
    labels = ["Fractal", "ESN", "LSTM", "GRU", "MLP"]
    vals = [mse_fm, mse_esn, mse_lstm, mse_gru, mse_mlp]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals)
    plt.ylabel("MSE")
    plt.title(f"{gen_label}: MSE comparison")
    plt.tight_layout()
    p2 = os.path.join(outdir, f"{gen_label}_mse_bar.png")
    plt.savefig(p2, dpi=150)
    plt.close()

    # Return results
    return {
        "label": gen_label,
        "mse_fractal": float(mse_fm),
        "mse_esn": float(mse_esn),
        "mse_lstm": float(mse_lstm),
        "mse_gru": float(mse_gru),
        "mse_mlp": float(mse_mlp),
    }


# -----------------------
# Main loop: iterate generators
# -----------------------
def main():
    results = []
    for label, fn, kwargs in GENERATORS:
        try:
            res = run_experiment(label, fn, kwargs)
            results.append(res)
        except Exception as e:
            print(f"ERROR running {label}: {e}")
            # Continue with others

    # Save CSV summary
    csv_path = os.path.join(OUTROOT, "summary_mse.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generator", "mse_fractal", "mse_esn", "mse_lstm", "mse_gru", "mse_mlp"])
        for r in results:
            writer.writerow([r["label"], r["mse_fractal"], r["mse_esn"], r["mse_lstm"], r["mse_gru"], r["mse_mlp"]])

    print("\n=== Completed all generators ===")
    print("Summary CSV:", csv_path)
    print("Per-generator plots in:", OUTROOT)


if __name__ == "__main__":
    main()


