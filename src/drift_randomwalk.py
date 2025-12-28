"""
src/drift_randomwalk.py
Random-walk (stochastic) parameter drift generator. Example: frequency performs small Gaussian random walk.

Run:
    python -m src.drift_randomwalk
"""
import numpy as np

def gen_randomwalk_freq_sine(T=3000, f0=0.02, sigma_f=1e-4, A=1.0, noise_std=0.01, seed=0):
    rng = np.random.RandomState(seed)
    f_t = np.empty(T, dtype=np.float32)
    f_t[0] = f0
    for t in range(1, T):
        f_t[t] = max(1e-6, f_t[t-1] + rng.randn() * sigma_f)
    t = np.arange(T)
    x = A * np.sin(2*np.pi * (f_t * t)) + noise_std * rng.randn(T)
    return x.astype(np.float32), f_t.astype(np.float32)


if __name__ == "__main__":
    print("Random-walk freq sine quick test")
    x, f_t = gen_randomwalk_freq_sine(T=1000, seed=3)
    print("shape:", x.shape, "freq sample:", f_t[:8])



