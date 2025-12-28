# synthetic signal generators
"""
src/data.py
Synthetic signal generators for fractal-memory experiments.

Functions:
- gen_drifting_sine(T=4000, seed=0): drifting-frequency & amplitude sine + noise
- gen_piecewise_freq_sine(T=2000, seed=0): piecewise frequency changes
- simple_test_plot(): quick visual test (used by __main__)
"""

import numpy as np
import matplotlib.pyplot as plt


def gen_drifting_sine(T=4000, seed=0):
    """Generate a drifting sine wave with slow amplitude & frequency modulations."""
    np.random.seed(seed)
    t = np.arange(T)
    freq = 0.01 + 0.02 * np.sin(0.0008 * t) + 0.005 * np.cos(0.0017 * t)
    amp = 1.0 + 0.2 * np.sin(0.0005 * t + 1.3)
    phase = 0.5 * np.cumsum(0.01 * np.sin(0.0002 * t))
    x = amp * np.sin(2 * np.pi * freq * t + phase) + 0.05 * np.random.randn(T)
    return x.astype(np.float32)


def gen_piecewise_freq_sine(T=2000, seed=0, n_segments=4):
    """Generate a sine where frequency jumps every segment (useful for abrupt drift tests)."""
    np.random.seed(seed)
    seg_len = T // n_segments
    x = np.zeros(T, dtype=np.float32)
    for i in range(n_segments):
        f = 0.005 + 0.02 * np.random.rand()
        amp = 0.8 + 0.4 * np.random.rand()
        start = i * seg_len
        end = start + seg_len if i < n_segments - 1 else T
        t = np.arange(end - start)
        x[start:end] = amp * np.sin(2 * np.pi * f * t + np.random.rand() * 2 * np.pi)
    x += 0.03 * np.random.randn(T)
    return x


def simple_test_plot():
    x = gen_drifting_sine(T=1000, seed=1)
    plt.figure(figsize=(8, 3))
    plt.plot(x, label="drifting sine")
    plt.title("Test: drifting sine (1000 steps)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Running data.py quick test...")
    simple_test_plot()
    print("Done.")
