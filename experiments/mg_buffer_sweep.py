"""
experiments/mg_buffer_sweep.py
Memory-capacity ablation: vary episodic buffer size on Mackey–Glass.
Measures slow consolidated predictor performance only.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.mackey_glass import gen_mackey_glass_sequence
from src.fractal_model import FractalModel
from src.utils import mse


# -----------------------
# Config
# -----------------------
T = 3000
TRAIN_RATIO = 0.7
RES_SIZE = 300
SEED = 5


def run_case(buf_size):
    # Generate data
    x = gen_mackey_glass_sequence(T=T, seed=SEED)
    split = int(TRAIN_RATIO * len(x))

    # Build model
    model = FractalModel(
        res_size=RES_SIZE,
        buffer_maxlen=buf_size,
        reservoir_kwargs={
            "spectral_radius": 0.95,
            "leak": 0.25,
            "seed": SEED,
        },
    )

    # Run model
    out = model.run_sequence(
        x,
        consolidate_every=350,
        min_consolidate_samples=50,
    )

    preds = out["preds_slow"]

    # ----------------------------------
    # Correct alignment + test-only MSE
    # ----------------------------------
    # preds[t] predicts x[t+1]
    y_true = x[split + 1 :]
    y_pred = preds[split : split + len(y_true)]

    # Remove NaNs (before first consolidation)
    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return mse(y_true, y_pred)


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("=== MG BUFFER SIZE ABLATION (FIXED) ===")

    sizes = [100, 300, 600, 900, 1200, 1500]
    mses = []

    for s in sizes:
        print(f"Testing buffer size = {s}")
        val = run_case(s)
        mses.append(val)
        print(f"  MSE = {val:.6e}")

    print("\nSizes:", sizes)
    print("MSEs :", mses)

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(sizes, mses, marker="o")
    plt.xlabel("Episodic Buffer Size")
    plt.ylabel("MSE (slow consolidated predictor)")
    plt.title("Mackey–Glass: Buffer Size Ablation")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
