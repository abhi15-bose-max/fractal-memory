"""
src/drift_fast_mg.py
Fast-drifting Mackey–Glass generator.

Beta(t) changes quickly using a higher modulation frequency.
This stresses SHORT timescale adaptation.

Run:
    python -m src.drift_fast_mg
"""

import numpy as np

def gen_fast_drift_mg(T=2000, tau=17, beta_base=0.2, beta_amp=0.1, gamma=0.1, n=10, seed=0):
    """
    Fast drifting Mackey–Glass:
    beta(t) = beta_base + beta_amp * sin( 2π * f_fast * t )

    Fast drift frequency = ~2 full drift cycles over the entire sequence.
    """
    np.random.seed(seed)
    total = T + tau + 1
    x = np.zeros(total, dtype=np.float32)
    x[:tau] = 1.2 + 0.2 * np.random.randn(tau)

    f_fast = 2.0 / T  # high drift speed (multiple oscillations)

    for t in range(tau, total - 1):
        x_tau = x[t - tau]
        beta_t = beta_base + beta_amp * np.sin(2 * np.pi * f_fast * (t - tau))
        dx = beta_t * x_tau / (1 + x_tau**n) - gamma * x[t]
        x[t + 1] = x[t] + dx

    return x[-T:].astype(np.float32)


if __name__ == "__main__":
    print("Fast-drifting Mackey–Glass quick test...")
    x = gen_fast_drift_mg(T=2000, seed=3)
    print("Generated shape:", x.shape)
    print("First 10 values:", x[:10])
    print("Test finished.")
