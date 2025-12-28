"""
Experiment:
Effect of slow consolidation on k-step forecasting (k=1..20)
Compared to reservoir-only baseline
Evaluated on fast-drift and slow-drift Mackey–Glass

Outputs:
- RMSE vs k plots
Saved to: results/plots/
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.reservoir import Reservoir
from src.fractal_model import FractalModel
from src.readout import ridge_regression_fit

# ✅ CORRECT imports
from src.drift_fast_mg import gen_fast_drift_mg
from src.drift_slow_mg import gen_slow_drift_mg


# -------------------------
# Config
# -------------------------
T = 2500
TRAIN_FRAC = 0.6
RES_SIZE = 300
K_MAX = 20
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

SAVE_DIR = "results/plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------------
# Utility: RMSE
# -------------------------
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


# -------------------------
# Core experiment
# -------------------------
def run_experiment(x):
    split = int(TRAIN_FRAC * len(x))
    x_train, x_test = x[:split], x[split:]

    # ---------------------
    # Reservoir-only
    # ---------------------
    res = Reservoir(in_size=1, res_size=RES_SIZE, seed=SEED)
    H, Y = [], []

    for t in range(len(x_train) - 1):
        h = res.step(x_train[t]).cpu().numpy()
        H.append(h)
        Y.append(x_train[t + 1])

    H = np.stack(H)
    Y = np.array(Y)

    w, b = ridge_regression_fit(H, Y)

    def esn_predict(x_t):
        h = res.step(x_t).cpu().numpy()
        return h @ w + b

    # ---------------------
    # Fractal Memory
    # ---------------------
    fm = FractalModel(
        res_size=RES_SIZE,
        buffer_maxlen=1200,
        online_lr=1e-4
    )
    fm.run_sequence(x_train, consolidate_every=200)

    # ---------------------
    # k-step sweep
    # ---------------------
    ks = np.arange(1, K_MAX + 1)
    rmse_esn = []
    rmse_fm = []

    for k in ks:
        pred_e, tgt_e = [], []
        pred_f, tgt_f = [], []

        for t in range(len(x_test) - k):
            pred_e.append(esn_predict(x_test[t]))
            pred_f.append(fm.step(x_test[t])[0])
            tgt_e.append(x_test[t + k])
            tgt_f.append(x_test[t + k])

        rmse_esn.append(rmse(np.array(pred_e), np.array(tgt_e)))
        rmse_fm.append(rmse(np.array(pred_f), np.array(tgt_f)))

    return ks, np.array(rmse_esn), np.array(rmse_fm)


# -------------------------
# Run: fast & slow drift
# -------------------------
print("Running fast-drift experiment...")
x_fast = gen_fast_drift_mg(T=T, seed=SEED)
k_fast, esn_fast, fm_fast = run_experiment(x_fast)

print("Running slow-drift experiment...")


x_slow = gen_slow_drift_mg(T=T, seed=SEED)
k_slow, esn_slow, fm_slow = run_experiment(x_slow)


# -------------------------
# Plotting
# -------------------------
def plot_result(k, esn, fm, title, fname):
    plt.figure(figsize=(7, 5))
    plt.plot(k, esn, marker="o", label="Reservoir-only")
    plt.plot(k, fm, marker="s", label="Fractal Memory (slow consolidation)")
    plt.xlabel("k-step forecast horizon")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=200)
    plt.close()


plot_result(
    k_fast,
    esn_fast,
    fm_fast,
    "Fast Drift Mackey–Glass: RMSE vs Forecast Horizon",
    "rmse_vs_k_fast_drift.png"
)

plot_result(
    k_slow,
    esn_slow,
    fm_slow,
    "Slow Drift Mackey–Glass: RMSE vs Forecast Horizon",
    "rmse_vs_k_slow_drift.png"
)


# -------------------------
# Console summary
# -------------------------
print("\n=== Summary (k=10) ===")
k_idx = 9  # k=10
improv_fast = 100 * (esn_fast[k_idx] - fm_fast[k_idx]) / esn_fast[k_idx]
improv_slow = 100 * (esn_slow[k_idx] - fm_slow[k_idx]) / esn_slow[k_idx]

print(f"Fast drift: RMSE reduction = {improv_fast:.2f}%")
print(f"Slow drift: RMSE reduction = {improv_slow:.2f}%")

print(f"\nPlots saved to: {SAVE_DIR}")
