# experiment: no slow consolidation
"""
Ablation: No slow consolidation vs with slow consolidation.

Saves:
 - results/plots/ablation_no_slow_comparison.png
 - prints summary MSEs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.data import gen_drifting_sine
from src.fractal_model import FractalModel
from src.utils import mse, seed_everything

# --------------------
# Config
# --------------------
OUTDIR = "results/plots"
os.makedirs(OUTDIR, exist_ok=True)

SEED = 123
T = 1200
RES_SIZE = 200
BUFFER_MAXLEN = 800
CONSOL_EVERY = 300
MIN_CONSOL_SAMPLES = 50

seed_everything(SEED)

# --------------------
# Generate sequence
# --------------------
x = gen_drifting_sine(T=T, seed=SEED)

# optional: inject an abrupt regime change to test recovery
# e.g., at t=360 flip amplitude/frequency subtly
inject_idx = 360
x_injected = x.copy()
x_injected[inject_idx:inject_idx+80] *= 1.5  # stronger amplitude segment to stress the model

# We'll use x_injected for both runs so comparison is fair
x_run = x_injected

# --------------------
# Helper runner (no consolidation)
# --------------------
def run_no_consolidation():
    m = FractalModel(
        res_size=RES_SIZE,
        reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": 42},
        buffer_maxlen=BUFFER_MAXLEN,
        online_lr=1e-4,
        use_ae=False,   # simple baseline without AE
    )
    Tlocal = len(x_run)
    preds_fast = np.zeros(Tlocal, dtype=np.float32)
    preds_slow = np.zeros(Tlocal, dtype=np.float32)

    for t in range(Tlocal - 1):
        y_fast, h = m.step(x_run[t])
        preds_fast[t] = y_fast
        # add to buffer but DO NOT consolidate
        m.add_to_buffer(h, x_run[t+1])

        # do fast online update
        h_t = np.array(h, dtype=np.float32)
        import torch
        m.readout.online_sgd_step(torch.tensor(h_t, dtype=torch.float32), float(x_run[t+1]), lr=m.online_lr)

        # intentionally skip consolidation
        # if you want to exactly match code path but disable effect, you can omit maybe_consolidate

        # slow pred stays as current fast pred because no slow_w created
        sp = m.slow_predict(h)
        preds_slow[t] = sp if sp is not None else preds_fast[t]

    preds_fast[-1] = preds_fast[-2]
    preds_slow[-1] = preds_slow[-2]
    return preds_fast, preds_slow, m

# --------------------
# Helper runner (with consolidation)
# --------------------
def run_with_consolidation():
    m = FractalModel(
        res_size=RES_SIZE,
        reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": 42},
        buffer_maxlen=BUFFER_MAXLEN,
        online_lr=1e-4,
        use_ae=False,   # baseline without AE so consolidation fits raw H -> Y
    )
    Tlocal = len(x_run)
    preds_fast = np.zeros(Tlocal, dtype=np.float32)
    preds_slow = np.zeros(Tlocal, dtype=np.float32)

    for t in range(Tlocal - 1):
        y_fast, h = m.step(x_run[t])
        preds_fast[t] = y_fast
        m.add_to_buffer(h, x_run[t+1])

        # online fast update
        import torch
        m.readout.online_sgd_step(torch.tensor(np.array(h, dtype=np.float32)), float(x_run[t+1]), lr=m.online_lr)

        if (t + 1) % CONSOL_EVERY == 0 and m.buffer.size() >= MIN_CONSOL_SAMPLES:
            m.maybe_consolidate(min_samples=MIN_CONSOL_SAMPLES, alpha=1e-3)

        sp = m.slow_predict(h)
        preds_slow[t] = sp if sp is not None else preds_fast[t]

    preds_fast[-1] = preds_fast[-2]
    preds_slow[-1] = preds_slow[-2]
    return preds_fast, preds_slow, m

# --------------------
# Run experiments
# --------------------
print("Running NO consolidation run...")
pf_no, ps_no, m_no = run_no_consolidation()
print("Running WITH consolidation run...")
pf_yes, ps_yes, m_yes = run_with_consolidation()

# --------------------
# Metrics
# --------------------
window = 50  # rolling window for plotting rolling MSE
def rolling_mse(y_true, y_pred, w):
    out = []
    for i in range(len(y_true)):
        lo = max(0, i-w+1)
        out.append(np.mean((y_true[lo:i+1] - y_pred[lo:i+1])**2))
    return np.array(out, dtype=np.float32)

rmse_fast_no = rolling_mse(x_run, pf_no, window)
rmse_fast_yes = rolling_mse(x_run, pf_yes, window)
rmse_slow_no = rolling_mse(x_run, ps_no, window)
rmse_slow_yes = rolling_mse(x_run, ps_yes, window)

# summary
print("\nSummary (mean MSE over full run):")
print(f"Fast (no consolidation): {mse(x_run, pf_no):.6f}")
print(f"Fast (with consolidation): {mse(x_run, pf_yes):.6f}")
print(f"Slow (no consolidation): {mse(x_run, ps_no):.6f}")
print(f"Slow (with consolidation): {mse(x_run, ps_yes):.6f}")

# Recovery time measure (after inject_idx): time to drop to within 1.2x baseline MSE
def recovery_time(y_true, y_pred, inject_idx, baseline_window=200, factor=1.2):
    baseline = np.mean((y_true[:baseline_window] - y_pred[:baseline_window])**2)
    thresh = baseline * factor
    for i in range(inject_idx, len(y_true)):
        cur = np.mean((y_true[inject_idx:i+1] - y_pred[inject_idx:i+1])**2)
        if cur <= thresh:
            return i - inject_idx
    return None

rec_fast_no = recovery_time(x_run, pf_no, inject_idx)
rec_fast_yes = recovery_time(x_run, pf_yes, inject_idx)
rec_slow_no = recovery_time(x_run, ps_no, inject_idx)
rec_slow_yes = recovery_time(x_run, ps_yes, inject_idx)

print("\nRecovery times (steps after injected perturbation):")
print(f"Fast no consolidation: {rec_fast_no}")
print(f"Fast with consolidation: {rec_fast_yes}")
print(f"Slow no consolidation: {rec_slow_no}")
print(f"Slow with consolidation: {rec_slow_yes}")

# --------------------
# Plot
# --------------------
plt.figure(figsize=(12, 8))

ax1 = plt.subplot(3, 1, 1)
plt.plot(x_run, label="signal", alpha=0.8)
plt.plot(pf_no, label="fast no consolidation", alpha=0.9)
plt.plot(ps_no, label="slow no consolidation", alpha=0.9)
plt.axvline(inject_idx, color="k", linestyle="--", alpha=0.4)
plt.legend()
plt.title("No consolidation (fast vs slow)")

ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(x_run, label="signal", alpha=0.5)
plt.plot(pf_yes, label="fast with consolidation", alpha=0.9)
plt.plot(ps_yes, label="slow with consolidation", alpha=0.9)
plt.axvline(inject_idx, color="k", linestyle="--", alpha=0.4)
plt.legend()
plt.title("With consolidation")

ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(rmse_fast_no, label="rmse fast no", alpha=0.9)
plt.plot(rmse_fast_yes, label="rmse fast yes", alpha=0.9)
plt.plot(rmse_slow_no, label="rmse slow no", alpha=0.9)
plt.plot(rmse_slow_yes, label="rmse slow yes", alpha=0.9)
plt.axvline(inject_idx, color="k", linestyle="--", alpha=0.4)
plt.legend()
plt.title("Rolling MSE (window=%d)" % window)

plt.tight_layout()
outpath = os.path.join(OUTDIR, "ablation_no_slow_comparison.png")
plt.savefig(outpath, dpi=150)
print("\nSaved plot to:", outpath)
plt.show()
