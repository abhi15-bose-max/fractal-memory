# experiment: reservoir only
"""
Ablation: Reservoir only baseline
- No online SGD updates
- No episodic buffer
- No consolidation
- Readout stays at random initialization

Compare reservoir-only predictions to:
    - Signal
    - Fast predictor (online SGD)
    - Slow predictor (consolidated ridge)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.data import gen_drifting_sine
from src.fractal_model import FractalModel
from src.utils import mse, seed_everything

# ----------------------------------
# CONFIG
# ----------------------------------
OUTDIR = "results/plots"
os.makedirs(OUTDIR, exist_ok=True)

SEED = 123
T = 1200
RES_SIZE = 200

seed_everything(SEED)

x = gen_drifting_sine(T=T, seed=SEED)

# ----------------------------------
# RUN RESERVOIR-ONLY MODEL
# ----------------------------------

def run_reservoir_only():
    """
    Reservoir only:
    - Readout initialized randomly
    - NO online SGD
    - NO buffer
    - NO consolidation
    """
    model = FractalModel(
        res_size=RES_SIZE,
        reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": 42},
        buffer_maxlen=1,
        online_lr=0.0,     # disable SGD
        use_ae=False,
    )

    Tlocal = len(x)
    preds = np.zeros(Tlocal, dtype=np.float32)

    for t in range(Tlocal - 1):
        y, h_np = model.step(x[t])
        preds[t] = y
        # NOTE: NO SGD UPDATE
        # NOTE: NO BUFFER USE
        # NOTE: NO CONSOLIDATION

    preds[-1] = preds[-2]
    return preds, model


print("Running reservoir-only baseline...")
preds_res, model_res = run_reservoir_only()

# ----------------------------------
# RUN FULL MODEL (for comparison)
# ----------------------------------

print("Running full model (fast + consolidation)...")

def run_full():
    m = FractalModel(
        res_size=RES_SIZE,
        reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": 42},
        buffer_maxlen=800,
        online_lr=1e-4,
        use_ae=False,
    )
    out = m.run_sequence(x, consolidate_every=300, min_consolidate_samples=50)
    return out["preds_fast"], out["preds_slow"], m

preds_fast, preds_slow, model_full = run_full()

# ----------------------------------
# Print Metrics
# ----------------------------------

print("\nMSE Scores:")
print(f"Reservoir only:         {mse(x, preds_res):.6f}")
print(f"Fast predictor only:     {mse(x, preds_fast):.6f}")
print(f"Slow predictor (ridge):  {mse(x, preds_slow):.6f}")

# ----------------------------------
# Plot
# ----------------------------------

plt.figure(figsize=(12, 6))
plt.plot(x, label="signal", alpha=0.8, linewidth=1.0)
plt.plot(preds_res, label="reservoir only", alpha=0.9)
plt.plot(preds_fast, label="fast predictor", alpha=0.9)
plt.plot(preds_slow, label="slow predictor", alpha=0.9)

plt.legend()
plt.title("Ablation: Reservoir Only vs Fast vs Slow Predictor")
plt.tight_layout()

outpath = os.path.join(OUTDIR, "ablation_reservoir_only.png")
plt.savefig(outpath, dpi=160)
print("\nSaved plot to:", outpath)
plt.show()
