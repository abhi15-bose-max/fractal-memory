"""
experiments/auto_compare_with_metrics.py

Automatically run baseline comparisons (FractalModel, ESN+ridge, LSTM, GRU, MLP)
across multiple signal generators under src/, compute extra metrics (MSE, MAE, SMAPE, DTW, Lag)
and save per-generator plots + a summary CSV.

Run:
    python -m experiments.auto_compare_with_metrics
"""

import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Import models / utils
from src.fractal_model import FractalModel
from src.readout import ridge_regression_fit
from src.reservoir import Reservoir
from src.utils import mse as mse_fn, seed_everything

# Import generators (ensure these modules exist)
from src.drift_fast_mg import gen_fast_drift_mg
from src.drift_medium_mg import gen_medium_drift_mg
from src.drift_slow_mg import gen_slow_drift_mg
from src.drift_multiscale import gen_multiscale_drift
from src.drift_piecewise import gen_piecewise
from src.drift_randomwalk import gen_randomwalk_freq_sine
from src.drift_am_fm import gen_am_fm
from src.drift_heteroskedastic import gen_heteroskedastic_signal
from src.drift_composite import gen_composite_signal

# optional standard MG if available
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

EPOCHS = 20    # lower to speed up if needed
BATCH = 64
LR = 1e-3
HID = 64

# Generators list
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
# Metrics
# -----------------------
def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def smape(y_true, y_pred, eps=1e-8):
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + eps)))

def dtw_distance(a, b):
    """
    Simple DTW distance (squared-error cost), pure numpy implementation.
    O(N*M) time and memory. For long sequences this can be slow.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    m = len(b)
    # For safety, if very long, downsample both to max_len ~800
    max_len = 800
    if n > max_len or m > max_len:
        idx_a = np.round(np.linspace(0, n - 1, min(n, max_len))).astype(int)
        idx_b = np.round(np.linspace(0, m - 1, min(m, max_len))).astype(int)
        a = a[idx_a]
        b = b[idx_b]
        n = len(a)
        m = len(b)

    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])  # squared-error accumulated

def estimate_lag(y_true, y_pred, max_lag=200):
    """
    Estimate lag (in samples) by cross-correlation peak.
    Returns signed lag (positive means y_pred lags behind y_true).
    We restrict search to +/- max_lag for stability.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    # normalize
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    corr = np.correlate(yt, yp, mode="full")  # length 2n-1
    center = len(corr) // 2
    lo = max(0, center - max_lag)
    hi = min(len(corr), center + max_lag + 1)
    sub = corr[lo:hi]
    shift = sub.argmax() + (lo - center)
    return int(shift)


# -----------------------
# Helpers
# -----------------------
def maybe_extract_series(x_or_tuple):
    """Some generators return (x, meta) — return x if so."""
    if isinstance(x_or_tuple, tuple) or isinstance(x_or_tuple, list):
        return x_or_tuple[0]
    return x_or_tuple

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


# -----------------------
# Lightweight model wrappers
# -----------------------
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

def train_window_model(model, train_loader, epochs=EPOCHS, lr=LR, device="cpu"):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model


# -----------------------
# Run one generator experiment
# -----------------------
def run_experiment(gen_label, gen_fn, gen_kwargs):
    t_start = time.time()
    print(f"\n--- Running: {gen_label} ---")
    raw = gen_fn(**gen_kwargs)
    x = maybe_extract_series(raw)
    n = len(x)
    n_train = int(TRAIN_RATIO * n)
    x_train = x[:n_train]
    x_test = x[n_train:]

    # windows
    X_train, Y_train = build_windows(x_train, WIN)
    X_test, Y_test = build_windows(x_test, WIN)

    # FRACTAL MODEL
    fm = FractalModel(
        res_size=RES_SIZE,
        reservoir_kwargs={"spectral_radius":0.95, "leak":0.25, "seed":SEED},
        buffer_maxlen=1600,
        online_lr=1e-4,
        device=DEVICE,
    )
    out = fm.run_sequence(x, consolidate_every=300, min_consolidate_samples=50)
    preds_slow = out["preds_slow"]
    pred_fm = preds_slow[n_train:]
    # ensure lengths match
    if len(pred_fm) > len(x_test):
        pred_fm = pred_fm[: len(x_test)]
    if len(pred_fm) < len(x_test):
        pred_fm = np.pad(pred_fm, (0, len(x_test) - len(pred_fm)), constant_values=np.nan)

    # ESN + ridge
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

    # PyTorch window dataset
    class WindowDataset(torch.utils.data.Dataset):
        def __init__(self, X, Y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.Y = torch.tensor(Y, dtype=torch.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.Y[i]

    train_ds = WindowDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    # LSTM
    lstm = LSTMNet(WIN, HID)
    lstm = train_window_model(lstm, train_loader, epochs=EPOCHS, lr=LR, device=DEVICE)
    lstm.eval()
    with torch.no_grad():
        pred_lstm = lstm(torch.tensor(X_test).to(DEVICE)).cpu().numpy()

    # GRU
    gru = GRUNet(WIN, HID)
    gru = train_window_model(gru, train_loader, epochs=EPOCHS, lr=LR, device=DEVICE)
    gru.eval()
    with torch.no_grad():
        pred_gru = gru(torch.tensor(X_test).to(DEVICE)).cpu().numpy()

    # MLP
    mlp = MLPNet(WIN, HID)
    mlp = train_window_model(mlp, train_loader, epochs=EPOCHS, lr=LR, device=DEVICE)
    mlp.eval()
    with torch.no_grad():
        pred_mlp = mlp(torch.tensor(X_test).to(DEVICE)).cpu().numpy()

    # Align (pred arrays may be shorter due to windowing)
    aligned_fm = np.asarray(pred_fm, dtype=np.float32)
    aligned_esn = safe_align(pred_esn, len(x_test), start_idx=0)
    aligned_lstm = safe_align(pred_lstm, len(x_test), start_idx=WIN)
    aligned_gru = safe_align(pred_gru, len(x_test), start_idx=WIN)
    aligned_mlp = safe_align(pred_mlp, len(x_test), start_idx=WIN)

    # Truncate NaN tails to common valid length for metrics
    def crop_valid(y):
        y = np.asarray(y, dtype=np.float32)
        # find first NaN from the end and crop
        if np.isnan(y).any():
            y = y[~np.isnan(y)]
        return y

    # We'll compute metrics on the overlapping valid segment length L
    candidates = [aligned_fm, aligned_esn, aligned_lstm, aligned_gru, aligned_mlp]
    valid_lengths = [len(crop_valid(c)) for c in candidates]
    L = min(len(x_test), *valid_lengths)
    y_true = x_test[:L]

    preds = {
        "Fractal": crop_valid(aligned_fm)[:L],
        "ESN": crop_valid(aligned_esn)[:L],
        "LSTM": crop_valid(aligned_lstm)[:L],
        "GRU": crop_valid(aligned_gru)[:L],
        "MLP": crop_valid(aligned_mlp)[:L],
    }

    # compute metrics
    metrics = {}
    for name, p in preds.items():
        if len(p) != L:
            # pad shorter preds with NaN -> then drop
            p = np.pad(p, (0, max(0, L - len(p))), constant_values=np.nan)[:L]
        # if all NaN fallback
        if np.isnan(p).all():
            metrics[name] = {"mse": np.nan, "mae": np.nan, "smape": np.nan, "dtw": np.nan, "lag": np.nan}
            continue
        # replace any remaining NaNs with last valid value (stationary padding)
        nan_mask = np.isnan(p)
        if nan_mask.any():
            # forward fill
            idxs = np.where(~nan_mask)[0]
            if len(idxs) == 0:
                p = np.zeros(L)
            else:
                for i in range(L):
                    if nan_mask[i]:
                        p[i] = p[idxs[0]] if i < idxs[0] else p[idxs[-1]]
        # metrics
        m_mse = mse_fn(y_true, p)
        m_mae = mae(y_true, p)
        m_smape = smape(y_true, p)
        m_dtw = dtw_distance(y_true, p)
        m_lag = estimate_lag(y_true, p)
        metrics[name] = {"mse": m_mse, "mae": m_mae, "smape": m_smape, "dtw": m_dtw, "lag": m_lag}

    # Save plots and a small metrics table
    outdir = os.path.join(OUTROOT, gen_label)
    os.makedirs(outdir, exist_ok=True)

    # Segment plot (visual)
    t0 = 50
    Tplot = min(800, L - t0) if (L - t0) > 50 else L
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[t0 : t0 + Tplot], label="true", color="k", alpha=0.6)
    for name, p in preds.items():
        pvis = p[:L]
        plt.plot(pvis[t0 : t0 + Tplot], label=name, linewidth=1)
    plt.legend()
    plt.title(f"{gen_label}: Test Segment Predictions")
    plt.tight_layout()
    seg_path = os.path.join(outdir, f"{gen_label}_segment.png")
    plt.savefig(seg_path, dpi=150)
    plt.close()

    # Bar chart: metrics (normalize for plotting)
    # We collect metric values into arrays and normalize by max for plotting clarity
    metric_names = ["mse", "mae", "smape", "dtw", "lag"]
    vals = {mn: np.array([metrics[nm][mn] if not np.isnan(metrics[nm][mn]) else np.nan for nm in preds.keys()]) for mn in metric_names}
    # For plotting, replace NaN with max*1.1 so they show clearly as bad
    norm_vals = {}
    for mn in metric_names:
        arr = vals[mn]
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            norm_vals[mn] = np.zeros_like(arr)
            continue
        mx = finite.max()
        if mx == 0:
            mx = 1.0
        arr_plot = np.where(np.isfinite(arr), arr / mx, 1.1)
        norm_vals[mn] = arr_plot

    # Create grouped bar chart: x axis models, groups are metrics
    labels_models = list(preds.keys())
    x_idx = np.arange(len(labels_models))
    width = 0.12
    plt.figure(figsize=(10, 4))
    for i, mn in enumerate(metric_names):
        plt.bar(x_idx + i * width, norm_vals[mn], width=width, label=mn.upper())
    plt.xticks(x_idx + width * 2, labels_models)
    plt.ylim(0, 1.2)
    plt.legend()
    plt.title(f"{gen_label}: Metrics (normalized for visual comparison)")
    plt.tight_layout()
    bar_path = os.path.join(outdir, f"{gen_label}_metrics_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()

    # Save raw metrics to text file
    metrics_txt = os.path.join(outdir, f"{gen_label}_metrics_table.txt")
    with open(metrics_txt, "w") as f:
        f.write("model\tmse\tmae\tsmape\tdtw\tlag\n")
        for nm in preds.keys():
            d = metrics[nm]
            f.write(f"{nm}\t{d['mse']:.6e}\t{d['mae']:.6e}\t{d['smape']:.6e}\t{d['dtw']:.6e}\t{d['lag']}\n")

    t_end = time.time()
    elapsed = t_end - t_start
    print(f"Completed {gen_label} in {elapsed:.1f}s. Metrics saved to: {outdir}")

    # Return summary dict
    summary = {"label": gen_label, "time_s": elapsed}
    for nm in preds.keys():
        d = metrics[nm]
        summary.update({
            f"{nm}_mse": float(d["mse"]),
            f"{nm}_mae": float(d["mae"]),
            f"{nm}_smape": float(d["smape"]),
            f"{nm}_dtw": float(d["dtw"]),
            f"{nm}_lag": int(d["lag"]),
        })
    return summary


# -----------------------
# Main loop
# -----------------------
def main():
    results = []
    for label, fn, kwargs in GENERATORS:
        try:
            res = run_experiment(label, fn, kwargs)
            results.append(res)
        except Exception as e:
            print(f"ERROR on {label}: {e}")

    # Save CSV summary - columns vary depending on models; create header dynamically
    if len(results) == 0:
        print("No results to save.")
        return

    # Build header from keys of first result
    header = sorted(results[0].keys())
    csv_path = os.path.join(OUTROOT, "summary_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            row = [r.get(k, "") for k in header]
            writer.writerow(row)

    print("\nAll done. Summary CSV:", csv_path)
    print("Per-generator plots in:", OUTROOT)


if __name__ == "__main__":
    main()
