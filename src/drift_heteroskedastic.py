"""
src/drift_heteroskedastic.py
Generates a base signal with time-varying noise scale and occasional outlier spikes.

Run:
    python -m src.drift_heteroskedastic
"""
import numpy as np

def gen_heteroskedastic_signal(T=3000,
                               base_signal=None,
                               base_noise=0.02,
                               noise_amp=0.5, 
                               noise_rate=None,
                               spike_prob=0.001,
                               spike_scale=3.0,
                               seed=0):
    rng = np.random.RandomState(seed)
    if base_signal is None:
        # default base: drifting sine
        from .drift_multiscale import gen_multiscale_drift
        base_signal = gen_multiscale_drift(T=T, seed=seed)

    t = np.arange(T)
    noise_rate = noise_rate or (1.0 / (T * 10.0))
    noise_scale = base_noise * (1.0 + noise_amp * np.sin(2*np.pi*noise_rate*t))
    noise = noise_scale * rng.randn(T)
    spikes = rng.rand(T) < spike_prob
    spike_vals = spikes * (spike_scale * rng.randn(T))
    x = base_signal + noise + spike_vals
    return x.astype(np.float32)


if __name__ == "__main__":
    print("Heteroskedastic quick test")
    x = gen_heteroskedastic_signal(T=1000, seed=5)
    print("shape:", x.shape, "spikes approx:", (x>3).sum())
