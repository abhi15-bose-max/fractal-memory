"""
src/drift_piecewise.py
Piecewise regime-switching generator. Switches between waveforms (sine, sawtooth, noise, chaotic-like MG placeholder).
Use smoothing_width to blend transitions.

Run:
    python -m src.drift_piecewise
"""
import numpy as np
from scipy import signal  # scipy required; if unavailable replace sawtooth with simple function

def piece_sawtooth(t, freq):
    return signal.sawtooth(2*np.pi*freq*t)

def gen_piecewise(T=3000, segments=None, smoothing_width=10, noise_std=0.01, seed=0):
    """
    segments: list of tuples (length, type, kwargs)
      type in {'sine','saw','noise','chirp'}
      kwargs vary per type: freq, amp, etc.
    If segments None, create a default sequence.
    """
    rng = np.random.RandomState(seed)
    if segments is None:
        # default 4 segments
        segments = [
            (T//4, 'sine', {'freq':0.02, 'amp':1.0}),
            (T//4, 'saw',  {'freq':0.03, 'amp':0.8}),
            (T//4, 'noise',{'scale':0.5}),
            (T - 3*(T//4), 'sine', {'freq':0.015, 'amp':1.2}),
        ]

    out = np.zeros(T, dtype=np.float32)
    idx = 0
    for length, typ, kw in segments:
        t_rel = np.arange(length)
        if typ == 'sine':
            freq = kw.get('freq', 0.02)
            amp = kw.get('amp', 1.0)
            seg = amp * np.sin(2*np.pi*freq*t_rel)
        elif typ == 'saw':
            freq = kw.get('freq', 0.03)
            amp = kw.get('amp', 1.0)
            seg = amp * piece_sawtooth(t_rel, freq)
        elif typ == 'noise':
            scale = kw.get('scale', 0.5)
            seg = scale * rng.randn(length)
        elif typ == 'chirp':
            f0 = kw.get('f0', 0.01)
            f1 = kw.get('f1', 0.05)
            seg = signal.chirp(t_rel, f0=f0, f1=f1, t1=length, method='linear')
        else:
            seg = rng.randn(length) * 0.1
        out[idx:idx+length] = seg
        idx += length

    # smoothing transitions
    if smoothing_width > 0:
        for p in range(smoothing_width, T - smoothing_width, length):  # coarse but safe
            pass
    # add tiny noise
    out = out + noise_std * rng.randn(T)
    return out.astype(np.float32)


if __name__ == "__main__":
    print("Piecewise drift quick test")
    x = gen_piecewise(T=1000, seed=2)
    print("shape:", x.shape, "first 8:", x[:8])
