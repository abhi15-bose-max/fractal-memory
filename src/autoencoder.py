"""
src/autoencoder.py
Upgraded Autoencoder supporting:
- raw h training
- latent-only training (by decoding z back to h_hat)
- unified encode/decode single + batch APIs
- recon error helper
- safer device handling

This AE is now fully compatible with AE-enabled fractal memory.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# deterministic seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------
# Basic Encoder
# ---------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        layers.append(nn.Linear(last, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# Basic Decoder
# ---------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim=32, hidden_dims=(64, 128)):
        super().__init__()
        layers = []
        last = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------
# Full AutoEncoder
# ---------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, enc_hidden=(128, 64), dec_hidden=(64, 128), device="cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        self.encoder = Encoder(input_dim, latent_dim, hidden_dims=enc_hidden).to(device)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dims=dec_hidden).to(device)

        self.to(device)

    # -----------------------------------------------------
    # Forward: x -> x_reconstructed
    # -----------------------------------------------------
    def forward(self, x):
        z = self.encoder(x)
        xr = self.decoder(z)
        return xr

    # -----------------------------------------------------
    # Public encode/decode helpers
    # -----------------------------------------------------
    def encode(self, x_np):
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_np, dtype=torch.float32, device=self.device)
            z = self.encoder(x_t)
            return z.cpu().numpy()

    def decode(self, z_np):
        self.eval()
        with torch.no_grad():
            z_t = torch.tensor(z_np, dtype=torch.float32, device=self.device)
            xr = self.decoder(z_t)
            return xr.cpu().numpy()

    def encode_single(self, x_np):
        return self.encode(x_np.reshape(1, -1))[0]

    def decode_single(self, z_np):
        return self.decode(z_np.reshape(1, -1))[0]

    # -----------------------------------------------------
    # Reconstruction error helper
    # -----------------------------------------------------
    def recon_error(self, x_np):
        """
        Computes MSE reconstruction error for single or batch.
        """
        xr = self.decode(self.encode(x_np))
        return float(np.mean((xr - x_np) ** 2))

    # -----------------------------------------------------
    # Train step
    # -----------------------------------------------------
    def train_on_batch(self, x_batch, optimizer, loss_fn=nn.MSELoss()):
        self.train()
        if not torch.is_tensor(x_batch):
            x_batch = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        optimizer.zero_grad()
        xr = self.forward(x_batch)
        loss = loss_fn(xr, x_batch)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    # -----------------------------------------------------
    # Fit on raw reservoir states
    # -----------------------------------------------------
    def fit(self, X, epochs=5, batch_size=64, lr=5e-4, weight_decay=1e-6, verbose=True):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        N = X.shape[0]

        for ep in range(1, epochs + 1):
            perm = np.random.permutation(N)
            losses = []
            for i in range(0, N, batch_size):
                xb = X[perm[i:i + batch_size]]
                losses.append(self.train_on_batch(xb, optimizer))

            if verbose:
                print(f"[AE] epoch {ep}/{epochs}, mean loss={np.mean(losses):.6f}")

    # -----------------------------------------------------
    # Fit using samples from episodic buffer (raw OR latent)
    # -----------------------------------------------------
    def fit_from_buffer(self, samples, epochs=3, batch_size=64, lr=5e-4, verbose=False):
        if len(samples) == 0:
            return False

        first = samples[0]

        # CASE 1: raw samples (h_np, target)
        if len(first) == 2 and isinstance(first[0], np.ndarray):
            X = np.stack([s[0] for s in samples], axis=0)
            self.fit(X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
            return True

        # CASE 2: latent samples (z, target, recon_err, ts)
        elif len(first) >= 4:
            Z = np.stack([s[0] for s in samples], axis=0)
            # Decode to pseudo-target H_hat for self-training
            H_hat = self.decode(Z)
            self.fit(H_hat, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
            return True

        return False

    # -----------------------------------------------------
    # Save/Load
    # -----------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        map_location = map_location or self.device
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)


# ---------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n*** Autoencoder upgraded test ***\n")

    try:
        from src.reservoir import Reservoir
        from src.data import gen_drifting_sine
    except Exception:
        Reservoir = None

    RES_SIZE = 150
    LATENT = 32
    N = 2000

    if Reservoir is not None:
        print("Generating realistic reservoir snapshots...")
        r = Reservoir(in_size=1, res_size=RES_SIZE, spectral_radius=0.95, leak=0.25, seed=123)
        x = gen_drifting_sine(T=N + 50, seed=10)

        H = []
        for t in range(N):
            H.append(r.step(x[t]).cpu().numpy())
        H = np.stack(H)
    else:
        print("Reservoir not available — using random data.")
        H = np.random.randn(N, RES_SIZE).astype(np.float32)

    ae = Autoencoder(RES_SIZE, LATENT)

    # Before training
    pre = ae.recon_error(H[:256])
    print("Recon MSE before:", pre)

    # Train
    ae.fit(H, epochs=5, batch_size=128, lr=5e-4, verbose=True)

    # After training
    post = ae.recon_error(H[:256])
    print("Recon MSE after:", post)

    # Save
    path = "models/ae_upgraded.pth"
    ae.save(path)
    print("Saved checkpoint to:", path)
    print("Done.\n")
