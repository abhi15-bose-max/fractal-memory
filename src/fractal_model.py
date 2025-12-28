"""
src/fractal_model.py
FractalModel -- a small orchestrator that ties reservoir, episodic buffer, autoencoder, and readout together.

Features:
- Optional AE compression of reservoir states before storage (use_ae=True)
- Two consolidation styles:
    * latent consolidation (fit ridge on z -> target)
    * decoded consolidation (decode z -> h_hat, fit ridge on h_hat -> target)
- AE mini-training from buffer prior to consolidation
- Soft transfer of slow (consolidated) weights into fast readout
- Backwards compatible when use_ae=False

Run quick demo:
    python -m src.fractal_model
"""

import time
import numpy as np
import torch

from .reservoir import Reservoir
from .episodic import EpisodicBuffer
from .readout import Readout, ridge_regression_fit

# Try importing Autoencoder (optional)
try:
    from .autoencoder import Autoencoder
except Exception:
    Autoencoder = None


class FractalModel:
    def __init__(
        self,
        res_size=300,
        reservoir_kwargs=None,
        buffer_maxlen=1000,
        device="cpu",
        online_lr=1e-4,
        use_ae=False,
        ae_latent_dim=32,
        ae_train_epochs=3,
        ae_batch_size=64,
        ae_lr=5e-4,
        slow_on_latent=True,
    ):
        """
        Parameters:
        - res_size: reservoir state dimension
        - reservoir_kwargs: dict passed to Reservoir
        - buffer_maxlen: episodic buffer capacity
        - device: 'cpu' or 'cuda'
        - online_lr: fast readout SGD lr
        - use_ae: whether to compress episodic traces using Autoencoder
        - ae_latent_dim: latent dimension if AE is used
        - ae_train_epochs, ae_batch_size, ae_lr: AE training hyperparams (per consolidation)
        - slow_on_latent: if True, slow predictor is fit in latent space (z->y).
                          if False and use_ae=True, decode z->h_hat and fit in reservoir space.
        """
        reservoir_kwargs = reservoir_kwargs or {}
        self.device = device
        self.reservoir = Reservoir(in_size=1, res_size=res_size, device=device, **reservoir_kwargs)
        self.readout = Readout(res_size=res_size, device=device)

        # Episodic buffer: default to raw mode, will switch to latent if use_ae=True
        store_mode = "latent" if use_ae else "raw"
        self.buffer = EpisodicBuffer(maxlen=buffer_maxlen, store_mode=store_mode)

        # AE
        self.use_ae = bool(use_ae)
        self.ae = None
        self.ae_latent_dim = ae_latent_dim
        self.ae_train_epochs = ae_train_epochs
        self.ae_batch_size = ae_batch_size
        self.ae_lr = ae_lr
        self.slow_on_latent = bool(slow_on_latent)

        if self.use_ae:
            if Autoencoder is None:
                raise ImportError("Autoencoder requested but src.autoencoder.Autoencoder not available.")
            # instantiate AE
            self.ae = Autoencoder(input_dim=res_size, latent_dim=ae_latent_dim, device=device)

        # Slow predictors:
        # - if slow_on_latent: slow_w_latent (dim latent) + slow_b_latent
        # - else: slow_w (dim reservoir) + slow_b
        self.slow_w = None
        self.slow_b = 0.0
        self.slow_w_latent = None
        self.slow_b_latent = 0.0

        self.online_lr = online_lr

    def step(self, x_t):
        """
        Input x_t (scalar). Steps the reservoir, returns:
        - y_fast: prediction from current readout (scalar)
        - h_np: reservoir state (numpy vector)
        """
        h = self.reservoir.step(float(x_t))
        with torch.no_grad():
            y_fast_t = float(self.readout(h).item())
        h_np = h.cpu().numpy().copy()
        return y_fast_t, h_np

    def add_to_buffer(self, h_np, y_target_next):
        """
        Adds either raw h or latent z depending on AE presence.
        If AE in use, it encodes and stores z with recon_error metadata.
        """
        if not self.use_ae:
            # old behavior: store raw (h_np, target)
            self.buffer.add_raw(h_np, y_target_next)
            return

        # encode h -> z and compute reconstruction error
        z = self.ae.encode(np.asarray(h_np, dtype=np.float32).reshape(1, -1))  # shape (1, latent)
        z0 = z[0]
        # reconstruct and compute MSE recon error
        h_rec = self.ae.decode(z)  # shape (1, res_size)
        recon_err = float(np.mean((h_rec[0].astype(np.float32) - np.asarray(h_np, dtype=np.float32)) ** 2))
        self.buffer.add_latent(z0, y_target_next, recon_err)

    def train_ae_on_buffer(self, max_samples=1024, epochs=None, batch_size=None, lr=None):
        """
        Train AE for a few epochs on a sample from the buffer.
        Useful to adapt encoder/decoder before consolidation.
        """
        if not self.use_ae or self.ae is None:
            return False
        epochs = epochs or self.ae_train_epochs
        batch_size = batch_size or self.ae_batch_size
        lr = lr or self.ae_lr

        samples = self.buffer.sample(min(max_samples, self.buffer.size()))
        if len(samples) == 0:
            return False
        # samples in latent mode are tuples (z, target, recon_err, ts) but we need raw h for AE training
        # If the buffer stores latents only, we cannot reconstruct raw h to train AE.
        # So we assume AE is trained on raw h snapshots stored elsewhere or we optionally keep raw_h on disk.
        # Simpler approach: if we do not have raw h in buffer, we skip AE training here.
        # However earlier design planned to call AE.fit_from_buffer(samples) where samples contain raw h.
        # To support that, we check whether the buffer items appear to be raw or latent.
        first = samples[0]
        # If first item has length 2 -> raw mode (h, target)
        if len(first) >= 2 and isinstance(first[0], np.ndarray) and len(first) == 2:
            # raw buffer: easy
            X = np.stack([it[0] for it in samples], axis=0)
            self.ae.fit(X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)
            return True
        else:
            # latent buffer: we DON'T have raw h to train AE.
            # To keep AE trainable, user should either:
            #  - keep additional raw buffer, or
            #  - rely on offline pretraining
            # For now, we skip if buffer only contains latents.
            return False

    def maybe_consolidate(self, min_samples=50, alpha=1e-3, sample_n=1024, soft_mix=0.2):
        """
        If buffer has enough samples, perform consolidation:
        - If use_ae and slow_on_latent: fit ridge on (Z, Y) and store slow_w_latent
        - If use_ae and not slow_on_latent: decode Z->H_hat and fit ridge on H_hat (reservoir dim)
        - If not use_ae: same as original (fit ridge on raw H)
        After fitting, if we obtained reservoir-dim slow weights (slow_w), we soft-update fast readout.
        """
        if self.buffer.size() < min_samples:
            return False

        # If AE is used, attempt AE mini-training on raw buffer (if available)
        try:
            self.train_ae_on_buffer()
        except Exception:
            pass  # ignore AE training failures

        # Collect samples for consolidation
        # Prefer high-recon samples if latent mode, else uniform
        if self.use_ae and self.buffer.store_mode == "latent":
            samples = self.buffer.sample_high_recon(min(sample_n, self.buffer.size()))
            # items are (z_np, target, recon_err, ts)
            Z = np.stack([it[0] for it in samples], axis=0).astype(np.float32)
            Y = np.array([it[1] for it in samples], dtype=np.float32)
            if self.slow_on_latent:
                # fit ridge on Z -> Y
                w_lat, b_lat = ridge_regression_fit(Z, Y, alpha=alpha)
                self.slow_w_latent = w_lat
                self.slow_b_latent = b_lat
                return True
            else:
                # decode Z -> H_hat and fit ridge on H_hat -> Y
                H_hat = self.ae.decode(Z).astype(np.float32)  # (N, res_size)
                w, b = ridge_regression_fit(H_hat, Y, alpha=alpha)
                self.slow_w = w
                self.slow_b = b
                # we have reservoir-dim slow weights: soft transfer to fast readout
                try:
                    self.readout.soft_update_from_ridge(self.slow_w, self.slow_b, mix=soft_mix)
                except Exception:
                    # fallback: directly set if soft_update not available
                    self.readout.set_weights_from_ridge(self.slow_w, self.slow_b)
                return True

        else:
            # No AE: buffer stores raw H -> use uniform sample
            samples = self.buffer.sample(min(sample_n, self.buffer.size()))
            # items are (h_np, target)
            H = np.stack([it[0] for it in samples], axis=0).astype(np.float32)
            Y = np.array([it[1] for it in samples], dtype=np.float32)
            w, b = ridge_regression_fit(H, Y, alpha=alpha)
            self.slow_w = w
            self.slow_b = b
            # soft-transfer to readout
            try:
                self.readout.soft_update_from_ridge(self.slow_w, self.slow_b, mix=soft_mix)
            except Exception:
                self.readout.set_weights_from_ridge(self.slow_w, self.slow_b)
            return True

    def slow_predict(self, h_np):
        """
        Predict using the consolidated slow predictor.
        - If slow_on_latent and slow_w_latent is set: encode h->z then z.dot(w_lat)+b_lat
        - If slow_w exists (reservoir-dim): use h_np.dot(slow_w)+slow_b
        - Otherwise return None
        """
        if self.use_ae and self.slow_w_latent is not None:
            z = self.ae.encode(np.asarray(h_np, dtype=np.float32).reshape(1, -1))[0]
            return float(float(np.dot(z, self.slow_w_latent) + self.slow_b_latent))
        if self.slow_w is not None:
            return float(np.dot(h_np, self.slow_w) + self.slow_b)
        return None

    def run_sequence(self, x_sequence, consolidate_every=400, min_consolidate_samples=50):
        """
        Run over x_sequence (1D numpy). Returns dict with predictions and metrics.
        - online SGD on fast readout each step
        - adds episodes to buffer (latent or raw)
        - consolidates periodically
        """
        T = len(x_sequence)
        preds_fast = np.zeros(T, dtype=np.float32)
        preds_slow = np.zeros(T, dtype=np.float32)
        recon_errs = []  # recon errors when using AE
        for t in range(T - 1):
            y_fast, h_np = self.step(x_sequence[t])
            preds_fast[t] = y_fast
            self.add_to_buffer(h_np, x_sequence[t + 1])

            # tiny online SGD step to keep fast readout reactive
            h_t = torch.tensor(h_np, dtype=torch.float32)
            y_true = torch.tensor([x_sequence[t + 1]], dtype=torch.float32)
            self.readout.online_sgd_step(h_t, y_true, lr=self.online_lr)

            # If AE used, collect latest recon error for monitoring (peek last buffer item)
            if self.use_ae and self.buffer.size() > 0 and self.buffer.store_mode == "latent":
                last = self.buffer.all()[-1]
                # last tuple: (z, target, recon_err, ts)
                recon_errs.append(last[2])

            # consolidation schedule
            if (t + 1) % consolidate_every == 0 and self.buffer.size() >= min_consolidate_samples:
                self.maybe_consolidate(min_samples=min_consolidate_samples)

            # compute slow pred if available
            sp = self.slow_predict(h_np)
            if sp is not None:
                preds_slow[t] = sp
            else:
                preds_slow[t] = preds_fast[t]

        # fill last value
        preds_fast[-1] = preds_fast[-2]
        preds_slow[-1] = preds_slow[-2]

        return {
            "preds_fast": preds_fast,
            "preds_slow": preds_slow,
            "buffer_size": self.buffer.size(),
            "recon_errs": np.array(recon_errs, dtype=np.float32),
        }


# ----------------------
# Quick demo / test
# ----------------------
if __name__ == "__main__":
    print("FractalModel quick end-to-end test (short run)...")
    from src.data import gen_drifting_sine
    import matplotlib.pyplot as plt

    x = gen_drifting_sine(T=1200, seed=3)

    # 1) No-AE baseline (original behavior)
    model_base = FractalModel(
        res_size=200,
        reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": 42},
        buffer_maxlen=800,
        online_lr=1e-4,
        use_ae=False,
    )
    out_base = model_base.run_sequence(x, consolidate_every=300, min_consolidate_samples=50)
    print("Base buffer size after run:", out_base["buffer_size"])

    # 2) AE-enabled run (if AE available)
    if Autoencoder is not None:
        print("\nRunning AE-enabled model (with latent consolidation)...")
        model_ae = FractalModel(
            res_size=200,
            reservoir_kwargs={"spectral_radius": 0.95, "leak": 0.25, "seed": 42},
            buffer_maxlen=800,
            online_lr=1e-4,
            use_ae=True,
            ae_latent_dim=32,
            ae_train_epochs=3,
            slow_on_latent=True,
        )
        out_ae = model_ae.run_sequence(x, consolidate_every=300, min_consolidate_samples=50)
        print("AE buffer size after run:", out_ae["buffer_size"])
        print("AE recon errors sample (first 20):", out_ae["recon_errs"][:20])

        # Plot comparison
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(x, label="signal")
        plt.plot(out_ae["preds_fast"], label="fast pred")
        plt.plot(out_ae["preds_slow"], label="slow pred")
        plt.legend()
        plt.title("AE-enabled: signal vs predictions")

        plt.subplot(2, 1, 2)
        if out_ae["recon_errs"].size > 0:
            plt.plot(out_ae["recon_errs"], label="recon_err (buffer peek)")
            plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        print("Autoencoder not available; skipping AE-enabled demo.")

    print("Test finished.")
