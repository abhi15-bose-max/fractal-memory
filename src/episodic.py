"""
src/episodic.py
Episodic buffer with optional latent storage, reconstruction error tracking,
and priority sampling for AE-based consolidation.

Supports two storage modes:
- "raw":   store (h_np, target)
- "latent": store (z_np, target, recon_error, timestamp)

New features:
- sample_high_recon(n): samples top-n items by reconstruction error (useful for anomalies)
- sample_recent(n): samples the most recent n items
"""

from collections import deque
import random
import numpy as np
import time


class EpisodicBuffer:
    def __init__(self, maxlen=1000, store_mode="raw"):
        """
        store_mode:
            "raw"    -> store (h_vector, target)
            "latent" -> store (z_vector, target, recon_error, timestamp)
        """
        assert store_mode in ("raw", "latent")
        self.store_mode = store_mode
        self.buf = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def add_raw(self, h_np, target):
        """Store raw reservoir state directly (old behavior)."""
        self.buf.append((np.asarray(h_np, dtype=np.float32), float(target)))

    def add_latent(self, z_np, target, recon_error):
        """
        Store latent vector + metadata.
        Internally stores:
            (z_np, target, recon_error, timestamp)
        """
        ts = int(time.time() * 1000)  # ms timestamp
        item = (
            np.asarray(z_np, dtype=np.float32),
            float(target),
            float(recon_error),
            ts,
        )
        self.buf.append(item)

    def add(self, item):
        """
        Compatibility: if user directly passes a tuple, store as-is.
        (Used for old reservoir-only mode.)
        """
        self.buf.append(item)

    def sample(self, n):
        """Uniform sample without replacement."""
        n = min(n, len(self.buf))
        return random.sample(list(self.buf), n)

    def sample_high_recon(self, n):
        """
        Return top-n items by reconstruction error.
        Only valid in latent mode.
        """
        if self.store_mode != "latent" or len(self.buf) == 0:
            return self.sample(n)

        # Sort by recon_error (descending)
        sorted_buf = sorted(self.buf, key=lambda it: it[2], reverse=True)
        return sorted_buf[: min(n, len(sorted_buf))]

    def sample_recent(self, n):
        """Return the most recent n items."""
        if n >= len(self.buf):
            return list(self.buf)
        return list(self.buf)[-n:]

    def all(self):
        return list(self.buf)

    def size(self):
        return len(self.buf)

    def is_full(self):
        return len(self.buf) == self.maxlen


# --------------------------------------------------
# Quick test
# --------------------------------------------------
if __name__ == "__main__":
    print("EpisodicBuffer extended test...")

    # ---------- RAW MODE ----------
    print("\n[1] RAW MODE TEST")
    buf_raw = EpisodicBuffer(maxlen=5, store_mode="raw")
    for i in range(10):
        buf_raw.add_raw(h_np=[i * 0.1], target=i + 1)

    print("Raw buffer size (should be 5):", buf_raw.size())
    print("Sample raw:", buf_raw.sample(3))

    # ---------- LATENT MODE ----------
    print("\n[2] LATENT MODE TEST")
    buf_lat = EpisodicBuffer(maxlen=8, store_mode="latent")
    for i in range(12):
        z = np.array([i * 0.5, i * 0.1], dtype=np.float32)
        target = i + 1
        recon_err = abs(np.sin(i)) * 0.1  # fake recon error for demo
        buf_lat.add_latent(z, target, recon_err)

    print("Latent buffer size (should be 8):", buf_lat.size())
    print("Recent 3:", buf_lat.sample_recent(3))

    print("High recon sample (top 3):")
    for it in buf_lat.sample_high_recon(3):
        print("  z=", it[0], "target=", it[1], "recon_error=", it[2])

    print("\nAll done.")
