"""
experiments/mg_slow_only_vs_full.py
Ablation: REMOVE FAST ONLINE LEARNING.
Compare slow-only vs full Fractal Memory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.mackey_glass import gen_mackey_glass_sequence
from src.fractal_model import FractalModel
from src.utils import mse, seed_everything

# -----------------------
# Config
# -----------------------
OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)

seed_everything(123)

T = 3000
RES_SIZE = 300
CONSOL_EVERY = 300
MIN_SAMPLES = 50

# -----------------------
# Data
# -----------------------
x = gen_mackey_glass_sequence(T=T, seed=8)

# -----------------------
# Slow-only model
# -----------------------
class SlowOnly(FractalModel):
    def run_sequence(self, x, consolidate_every=300, min_consolidate_samples=50):
        T = len(x)
        preds_fast = np.zeros(T)
        preds_slow = np.zeros(T)

        for t in range(T - 1):
            _, h = super().step(x[t])
            preds_fast[t] = 0.0  # meaningless

            self.add_to_buffer(h, x[t + 1])

            # ❌ NO FAST SGD

            if (t + 1) % consolidate_every == 0:
                self.maybe_consolidate(min_samples=min_consolidate_samples)

            preds_slow[t] = (
                self.slow_predict(h) if self.slow_w is not None else 0.0
            )

        preds_slow[-1] = preds_slow[-2]
        return preds_slow


print("Running slow-only model...")
slow_only_model = SlowOnly(res_size=RES_SIZE)
pred_slow_only = slow_only_model.run_sequence(
    x,
    consolidate_every=CONSOL_EVERY,
    min_consolidate_samples=MIN_SAMPLES,
)

# -----------------------
# Full model
# -----------------------
print("Running full Fractal Memory model...")
full_model = FractalModel(
    res_size=RES_SIZE,
    online_lr=1e-4,
    buffer_maxlen=1500,
)

out_full = full_model.run_sequence(
    x,
    consolidate_every=CONSOL_EVERY,
    min_consolidate_samples=MIN_SAMPLES,
)

pred_fast_full = out_full["preds_fast"]
pred_slow_full = out_full["preds_slow"]

# -----------------------
# Metrics
# -----------------------
print("\n=== MSE Comparison (Mackey–Glass) ===")
print(f"Slow-only (slow predictor): {mse(x, pred_slow_only):.6f}")
print(f"Full model (fast):          {mse(x, pred_fast_full):.6f}")
print(f"Full model (slow):          {mse(x, pred_slow_full):.6f}")

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(12, 5))
plt.plot(x[:800], label="MG true", alpha=0.5)
plt.plot(pred_slow_only[:800], label="slow-only", linewidth=2)
plt.plot(pred_fast_full[:800], label="full (fast)", alpha=0.9)
plt.plot(pred_slow_full[:800], label="full (slow)", alpha=0.9)

plt.legend()
plt.title("Ablation: Slow-Only vs Full Fractal Memory")
plt.tight_layout()

outpath = os.path.join(OUT, "mg_slow_only_vs_full.png")
plt.savefig(outpath, dpi=150)
plt.show()

print("\nSaved plot to:", outpath)
