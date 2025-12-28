"""
experiments/mg_ablation_no_consolidation_vs_full.py

Ablation: Effect of slow consolidation on Mackey–Glass prediction.

Compares:
1) Full Fractal Memory (fast + buffer + consolidation)
2) No consolidation (fast + buffer only)

Outputs:
- MSE values
- Prediction comparison plot
"""

import numpy as np
import matplotlib.pyplot as plt
from src.mackey_glass import gen_mackey_glass_sequence
from src.fractal_model import FractalModel
from src.utils import mse, seed_everything


# -----------------------
# Config
# -----------------------
seed_everything(123)

T = 3000
RES_SIZE = 300
BUFFER_MAXLEN = 1500
CONSOL_EVERY = 300
MIN_SAMPLES = 50

OUTDIR = "results/plots"

# -----------------------
# Data
# -----------------------
x = gen_mackey_glass_sequence(T=T, seed=4)


# -----------------------
# Model variants
# -----------------------
class Fractal_NoConsolidation(FractalModel):
    def maybe_consolidate(self, *args, **kwargs):
        return False  # disable slow learning entirely


# -----------------------
# Run: No consolidation
# -----------------------
print("Running NO consolidation model...")

model_no = Fractal_NoConsolidation(
    res_size=RES_SIZE,
    reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": 1},
    buffer_maxlen=BUFFER_MAXLEN,
    online_lr=1e-4,
)

out_no = model_no.run_sequence(
    x,
    consolidate_every=10**9,   # effectively off
)

pred_fast_no = out_no["preds_fast"]
pred_slow_no = out_no["preds_slow"]  # identical to fast


# -----------------------
# Run: Full model
# -----------------------
print("Running FULL Fractal Memory model...")

model_full = FractalModel(
    res_size=RES_SIZE,
    reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": 1},
    buffer_maxlen=BUFFER_MAXLEN,
    online_lr=1e-4,
)

out_full = model_full.run_sequence(
    x,
    consolidate_every=CONSOL_EVERY,
    min_consolidate_samples=MIN_SAMPLES,
)

pred_fast_full = out_full["preds_fast"]
pred_slow_full = out_full["preds_slow"]


# -----------------------
# Metrics
# -----------------------
print("\n=== MSE Comparison ===")
print(f"No consolidation (fast): {mse(x, pred_fast_no):.6f}")
print(f"Full model (fast):        {mse(x, pred_fast_full):.6f}")
print(f"Full model (slow):        {mse(x, pred_slow_full):.6f}")


# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(12, 4))
N = 800

plt.plot(x[:N], label="MG true", color="black", alpha=0.6)
plt.plot(pred_fast_no[:N], label="Fast (no consolidation)", alpha=0.9)
plt.plot(pred_slow_full[:N], label="Slow (with consolidation)", alpha=0.9)

plt.legend()
plt.title("Effect of Slow Consolidation on Mackey–Glass Prediction")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/mg_ablation_no_consolidation_vs_full.png", dpi=150)
plt.show()
