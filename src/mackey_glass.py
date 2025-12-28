"""
src/mackey_glass.py

Mackey–Glass chaotic time-series generator.
Useful as a nonlinear benchmark for reservoir computing and multi-timescale memory models.

Functions:
- mackey_glass(T, tau, beta, gamma, n, seed): generate raw series
- gen_mackey_glass_sequence(T, seed): convenient normalized wrapper

Includes a quick test at the bottom (run: python -m src.mackey_glass).
"""

import numpy as np


def mackey_glass(T=2000, tau=17, beta=0.2, gamma=0.1, n=10, seed=42):
    """
    Generate Mackey–Glass time series using Euler integration.

    Parameters:
    - T : length of time series
    - tau : delay parameter (classic MG uses tau=17)
    - beta, gamma, n : MG equation parameters
    - seed : reproducibility

    Returns:
    - x : numpy array of shape (T,)
    """

    np.random.seed(seed)

    # total integration steps
    total = T + tau + 1
    x = np.zeros(total, dtype=np.float32)

    # initialize history with random values
    x[:tau] = 1.2 + 0.2 * np.random.randn(tau)

    # Euler integration loop
    for t in range(tau, total - 1):
        x_tau = x[t - tau]   # delayed term
        dx = beta * x_tau / (1.0 + x_tau**n) - gamma * x[t]
        x[t + 1] = x[t] + dx

    # return the last T samples
    return x[-T:].astype(np.float32)


def gen_mackey_glass_sequence(T=2000, seed=42):
    """
    Convenient wrapper that:
    - generates Mackey–Glass series
    - normalizes it to zero-mean, unit-variance
    """
    x = mackey_glass(T=T, seed=seed)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x.astype(np.float32)


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    print("Mackey–Glass quick test...")

    T = 2000
    x = gen_mackey_glass_sequence(T=T, seed=123)

    print("Generated MG series shape:", x.shape)
    print("Mean (should be ~0):", float(x.mean()))
    print("Std  (should be ~1):", float(x.std()))

    # show first 10 values to confirm it's not constant
    print("First 10 values:", x[:10])

    try:
        import matplotlib.pyplot as plt
        plt.plot(x[:500], label="MG chaotic segment")
        plt.title("Mackey–Glass Quick Test")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        print("matplotlib not available, skipping plot.")

    print("Mackey–Glass test finished.")
