"""
src/drift_composite.py
Composite signal: mix of chaotic Mackey–Glass + drifting sine + noise.

Run:
    python -m src.drift_composite
"""
import numpy as np

def gen_composite_signal(T=3000,
                         mg_weight=0.6,
                         sine_weight=0.4,
                         mg_params=None,
                         sine_params=None,
                         noise_std=0.01,
                         seed=0):
    rng = np.random.RandomState(seed)
    # lazy imports to avoid circular issues
    try:
        from .drift_slow_mg import gen_slow_drift_mg
    except Exception:
        # fallback: generate random chaotic-like signal
        def gen_slow_drift_mg(T, seed=0, **k):
            rng2 = np.random.RandomState(seed)
            return (rng2.randn(T).cumsum() % 3) * 0.5
    try:
        from .drift_multiscale import gen_multiscale_drift
    except Exception:
        def gen_multiscale_drift(T, seed=0, **k):
            rng2 = np.random.RandomState(seed)
            return np.sin(np.linspace(0, 50, T)) * (1 + 0.1 * rng2.randn(T))

    mg = gen_slow_drift_mg(T=T, seed=seed, **(mg_params or {}))
    s = gen_multiscale_drift(T=T, seed=seed+1, **(sine_params or {}))
    x = mg_weight * mg + sine_weight * s + noise_std * rng.randn(T)
    return x.astype(np.float32)


if __name__ == "__main__":
    print("Composite quick test")
    x = gen_composite_signal(T=1000, seed=6)
    print("shape:", x.shape, "first 8:", x[:8])
