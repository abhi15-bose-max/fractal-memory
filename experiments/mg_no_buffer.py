"""
experiments/mg_no_buffer.py
Ablation: REMOVE EPISODIC BUFFER (no replay)
Compare against full Fractal Memory model.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.mackey_glass import gen_mackey_glass_sequence
from src.fractal_model import FractalModel
from src.utils import mse


# -------------------------------------------------
# No-buffer ablated model
# -------------------------------------------------
class NoBufferModel(FractalModel):
    def add_to_buffer(self, *args, **kwargs):
        return  # disable buffer entirely

    def maybe_consolidate(self, *args, **kwargs):
        return False  # disable slow consolidation


# -------------------------------------------------
# Main experiment
# -------------------------------------------------
if __name__ == "__main__":
    print("=== MG Ablation: NO EPISODIC BUFFER ===")

    T = 3000
    SEED = 3

    x = gen_mackey_glass_sequence(T=T, seed=SEED)

    # -------------------------
    # No-buffer model
    # -------------------------
    model_nb = NoBufferModel(
        res_size=300,
        reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": SEED},
        buffer_maxlen=1,
        online_lr=1e-4,
    )

    out_nb = model_nb.run_sequence(x, consolidate_every=10**9)
    fast_nb = out_nb["preds_fast"]
    slow_nb = out_nb["preds_slow"]  # identical to fast

    # -------------------------
    # Full Fractal Memory model
    # -------------------------
    model_full = FractalModel(
        res_size=300,
        reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": SEED},
        buffer_maxlen=1500,
        online_lr=1e-4,
    )

    out_full = model_full.run_sequence(
        x,
        consolidate_every=300,
        min_consolidate_samples=50
    )

    fast_full = out_full["preds_fast"]
    slow_full = out_full["preds_slow"]

    # -------------------------
    # Print MSEs
    # -------------------------
    print("\nMSE comparison (Mackey–Glass):")
    print(f"No buffer (fast only):   {mse(x, fast_nb):.6f}")
    print(f"Full model (fast):       {mse(x, fast_full):.6f}")
    print(f"Full model (slow):       {mse(x, slow_full):.6f}")

    # -------------------------
    # Plot comparison
    # -------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(x[:800], label="MG true", alpha=0.5)
    plt.plot(fast_nb[:800], label="no buffer (fast)", linewidth=1.5)
    plt.plot(slow_full[:800], label="full model (slow)", linewidth=1.5)
    plt.legend()
    plt.title("MG Ablation: No Episodic Buffer vs Full Model")
    plt.tight_layout()
    plt.show()
