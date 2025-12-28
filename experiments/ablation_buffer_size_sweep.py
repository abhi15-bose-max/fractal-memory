"""
Buffer size sweep: effect of episodic buffer capacity on consolidated (slow) MSE
on drifting sine.
"""

import os, csv, numpy as np
import matplotlib.pyplot as plt

from src.data import gen_drifting_sine
from src.fractal_model import FractalModel
from src.utils import mse, seed_everything

# -----------------------
# Config
# -----------------------
OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)

seed_everything(123)

T = 1200
TRAIN_RATIO = 0.7
RES_SIZE = 200

buffer_sizes = [50, 200, 800, 2000]

# -----------------------
# Data
# -----------------------
x = gen_drifting_sine(T=T, seed=123)
split = int(TRAIN_RATIO * len(x))

x_test = x[split:]

results = []

# -----------------------
# Sweep
# -----------------------
for bs in buffer_sizes:
    print("Running buffer size:", bs)

    model = FractalModel(
        res_size=RES_SIZE,
        reservoir_kwargs={"seed": 42},
        buffer_maxlen=bs,
        online_lr=1e-4,
        use_ae=False,
    )

    out = model.run_sequence(
        x,
        consolidate_every=300,
        min_consolidate_samples=50,
    )

    preds_fast = out["preds_fast"]
    preds_slow = out["preds_slow"]

    # ----------------------------------
    # Correct alignment (predict x[t+1])
    # ----------------------------------
    y_true = x_test[1:]
    y_fast = preds_fast[split : split + len(y_true)]
    y_slow = preds_slow[split : split + len(y_true)]

    # Remove NaNs (pre-consolidation)
    mask_fast = ~np.isnan(y_fast)
    mask_slow = ~np.isnan(y_slow)

    mse_fast = mse(y_true[mask_fast], y_fast[mask_fast])
    mse_slow = mse(y_true[mask_slow], y_slow[mask_slow])

    results.append({
        "buffer_size": bs,
        "mse_fast": float(mse_fast),
        "mse_slow": float(mse_slow),
        "final_buffer": int(out["buffer_size"]),
    })

# -----------------------
# Save CSV
# -----------------------
csvpath = os.path.join(OUT, "buffer_size_sweep.csv")
with open(csvpath, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["buffer_size", "mse_fast", "mse_slow", "final_buffer"],
    )
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print("Saved CSV:", csvpath)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(6, 4))
bs = [r["buffer_size"] for r in results]
mse_slow = [r["mse_slow"] for r in results]
mse_fast = [r["mse_fast"] for r in results]

plt.plot(bs, mse_fast, marker="o", label="Fast readout (online)")
plt.plot(bs, mse_slow, marker="o", label="Slow consolidated readout")

plt.xscale("log")
plt.xlabel("Episodic buffer size (log scale)")
plt.ylabel("MSE (test segment)")
plt.title("Effect of Episodic Buffer Capacity")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(OUT, "buffer_size_sweep.png"), dpi=150)
plt.show()
