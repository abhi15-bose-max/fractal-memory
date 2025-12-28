"""
src/drift_slow_mg.py
Slow-drifting Mackey–Glass generator.

Beta(t) drifts extremely slowly over time.
Tests LONG timescale memory (slow consolidation layer).

Run:
    python -m src.drift_slow_mg
"""

import numpy as np

def gen_slow_drift_mg(T=2000, tau=17, beta_base=0.2, beta_amp=0.1, gamma=0.1, n=10, seed=0):
    """
    Slow drifting Mackey–Glass:
    beta(t) = beta_base + beta_amp * sin(2π * f_slow * t)

    Very slow drift frequency = ~0.1 cycles over entire T.
    """
    np.random.seed(seed)
    total = T + tau + 1
    x = np.zeros(total, dtype=np.float32)
    x[:tau] = 1.2 + 0.2 * np.random.randn(tau)

    f_slow = 0.1 / T  # very slow drift (one 10th cycle)

    for t in range(tau, total - 1):
        x_tau = x[t - tau]
        beta_t = beta_base + beta_amp * np.sin(2 * np.pi * f_slow * (t - tau))
        dx = beta_t * x_tau / (1 + x_tau**n) - gamma * x[t]
        x[t + 1] = x[t] + dx

    return x[-T:].astype(np.float32)


if __name__ == "__main__":
    print("Slow-drifting Mackey–Glass quick test...")
    x = gen_slow_drift_mg(T=2000, seed=11)
    print("Generated shape:", x.shape)
    print("First 10 values:", x[:10])
    print("Test finished.")
