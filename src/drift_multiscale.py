"""
src/drift_multiscale.py
Multi-scale drifting sine: slow + fast modulation of frequency and amplitude.

Run:
    python -m src.drift_multiscale
"""
import numpy as np

def gen_multiscale_drift(T=3000,
                         f0=0.02,
                         f_slow_amp=0.005,
                         f_fast_amp=0.01,
                         f_slow_rate=None,
                         f_fast_rate=None,
                         A0=1.0,
                         A_mod=0.3,
                         noise_std=0.02,
                         seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(T).astype(np.float32)

    # sensible defaults: slow is very slow, fast is moderate
    f_slow_rate = f_slow_rate or (1.0 / (T * 5.0))
    f_fast_rate = f_fast_rate or (1.0 / (T / 20.0))

    f_t = f0 + f_slow_amp * np.sin(2*np.pi*f_slow_rate*t) + f_fast_amp * np.sin(2*np.pi*f_fast_rate*t)
    A_t = A0 + A_mod * np.sin(2*np.pi*(1.0/(T*10.0))*t)
    x = A_t * np.sin(2*np.pi * f_t * t) + noise_std * rng.randn(T)
    return x.astype(np.float32)


if __name__ == "__main__":
    print("Multiscale drift quick test")
    x = gen_multiscale_drift(T=1000, seed=1)
    print("shape:", x.shape, "first 8:", x[:8])
