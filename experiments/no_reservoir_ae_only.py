"""
No-reservoir baseline:
- No Reservoir, no fast readout.
- Build dataset from sliding windows of raw signal (window_len -> input vector).
- Train AE on windows (offline warmup).
- Store latents in an episodic buffer (simulated).
- Fit ridge on Z -> next_value (slow predictor).
- Evaluate next-step MSE.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from src.data import gen_drifting_sine
from src.autoencoder import Autoencoder
from src.readout import ridge_regression_fit
from src.utils import mse, seed_everything

OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)

seed_everything(123)
T = 1200
x = gen_drifting_sine(T=T, seed=123)

# sliding window parameters
win = 20  # input dimension for AE
latent_dim = 16

# build dataset of windows and next-step targets
X = []
Y = []
for t in range(T - win - 1):
    X.append(x[t : t + win])
    Y.append(x[t + win])
X = np.stack(X).astype(np.float32)
Y = np.array(Y).astype(np.float32)

# AE: train offline on all windows (small model)
ae = Autoencoder(input_dim=win, latent_dim=latent_dim, enc_hidden=(64,32), dec_hidden=(32,64), device="cpu")
ae.fit(X, epochs=8, batch_size=128, lr=5e-4, verbose=True)

# encode all windows
Z = ae.encode(X)  # (N, latent_dim)
w, b = ridge_regression_fit(Z, Y, alpha=1e-3)

# evaluate on whole series (predicting next-step from windows)
preds = (Z @ w) + b
print("No-reservoir AE-only MSE:", mse(Y, preds))

# plot a window of results
plt.figure(figsize=(10,4))
t0 = 300
plt.plot(range(t0, t0+200), Y[t0:t0+200], label="true next")
plt.plot(range(t0, t0+200), preds[t0:t0+200], label="pred from AE+ridge")
plt.legend()
plt.title("No-reservoir (AE only) next-step prediction")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "no_reservoir_ae_only.png"), dpi=150)
plt.show()



