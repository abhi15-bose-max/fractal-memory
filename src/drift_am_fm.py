"""
src/drift_am_fm.py
Amplitude + Frequency modulation combined generator.

Run:
    python -m src.drift_am_fm
"""
import numpy as np

def gen_am_fm(T=3000,
              base_freq=0.02,
              fm_amp=0.005,
              fm_rate=None,
              base_amp=1.0,
              am_amp=0.3,
              am_rate=None,
              noise_std=0.01,
              seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    fm_rate = fm_rate or (1.0 / (T * 10.0))
    am_rate = am_rate or (1.0 / (T * 15.0))
    f_t = base_freq + fm_amp * np.sin(2*np.pi*fm_rate*t)
    A_t = base_amp + am_amp * np.sin(2*np.pi*am_rate*t)
    x = A_t * np.sin(2*np.pi * f_t * t) + noise_std * rng.randn(T)
    return x.astype(np.float32)


if __name__ == "__main__":
    print("AM+FM quick test")
    x = gen_am_fm(T=1000, seed=4)
    print("shape:", x.shape, "first 8:", x[:8])
