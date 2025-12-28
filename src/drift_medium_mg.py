"""
src/drift_medium_mg.py
Medium-drifting Mackey–Glass generator.

Beta(t) drifts at a moderate rate.
This tests MID timescale memory (episodic buffer).

Run:
    python -m src.drift_medium_mg
"""

import numpy as np

def gen_medium_drift_mg(T=2000, tau=17, beta_base=0.2, beta_amp=0.1, gamma=0.1, n=10, seed=0):
    """
    Medium drifting Mackey–Glass:
    beta(t) = beta_base + beta_amp * sin(2π * f_mid * t)

    Medium drift frequency = ~0.5 cycle over T (slow but noticeable).
    """
    np.random.seed(seed)
    total = T + tau + 1
    x = np.zeros(total, dtype=np.float32)
    x[:tau] = 1.2 + 0.2 * np.random.randn(tau)

    f_mid = 0.5 / T  # slower drift (half oscillation over T)

    for t in range(tau, total - 1):
        x_tau = x[t - tau]
        beta_t = beta_base + beta_amp * np.sin(2 * np.pi * f_mid * (t - tau))
        dx = beta_t * x_tau / (1 + x_tau**n) - gamma * x[t]
        x[t + 1] = x[t] + dx

    return x[-T:].astype(np.float32)


if __name__ == "__main__":
    print("Medium-drifting Mackey–Glass quick test...")
    x = gen_medium_drift_mg(T=2000, seed=7)
    print("Generated shape:", x.shape)
    print("First 10 values:", x[:10])
    print("Test finished.")
