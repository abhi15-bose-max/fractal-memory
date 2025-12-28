"""
src/readout.py
Fast readout head and consolidation utilities.

- Readout (torch.nn.Module): linear readout from reservoir state to scalar prediction
- ridge_regression_fit(H, Y, alpha): small numpy ridge solver for consolidation
- Utility helpers: predict_batch_numpy, get/set weights, soft-update from ridge weights
- Example usage in __main__ demonstrates ridge fitting and soft-transfer of slow weights.
"""

import os
import numpy as np
import torch
import torch.nn as nn


class Readout(nn.Module):
    def __init__(self, res_size, device="cpu"):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(res_size, 1, bias=True).to(device)

    def forward(self, h):
        """
        h: torch tensor of shape (res_size,) or (batch, res_size)
        returns: 1-D tensor of shape (batch,) even for single input (batch=1)
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)               # -> (1, D)
        out = self.linear(h).squeeze(-1)     # -> (batch,)
        return out

    def predict_numpy(self, h_np):
        """Take numpy array (D,) and return scalar prediction (float)."""
        h_t = torch.tensor(h_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.forward(h_t)         # (1,)
            return float(out.squeeze(0).cpu().item())

    def predict_batch_numpy(self, H_np):
        """
        Predict on batch of numpy vectors H_np shape (N, D). Returns numpy array (N,).
        """
        H_t = torch.tensor(H_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.forward(H_t)        # (N,)
            return out.cpu().numpy()

    def online_sgd_step(self, h_vec, y_true, lr=1e-4):
        """
        h_vec: torch tensor shape (res_size,) or (1, res_size)
        y_true: scalar tensor or shape (1,) or python float
        performs single-step SGD (in-place) on the linear layer
        Returns: loss value (float)
        """
        self.train()
        # normalize h_vec to shape (1, D)
        if h_vec.dim() == 1:
            h_vec = h_vec.unsqueeze(0)

        # compute prediction (shape (1,))
        y_pred = self.forward(h_vec)

        # normalize y_true to same shape as y_pred
        if not torch.is_tensor(y_true):
            y_true = torch.tensor(float(y_true), dtype=torch.float32, device=y_pred.device)
        else:
            y_true = y_true.to(dtype=torch.float32, device=y_pred.device)

        # If y_true is scalar (0-d), expand to match y_pred (1-d)
        if y_true.dim() == 0 and y_pred.dim() == 1:
            y_true = y_true.unsqueeze(0)
        if y_pred.dim() == 0 and y_true.dim() == 1:
            y_pred = y_pred.unsqueeze(0)

        # ensure shapes match now
        if y_pred.shape != y_true.shape:
            try:
                y_true = y_true.view_as(y_pred)
            except Exception:
                raise ValueError(f"Incompatible shapes for loss: y_pred {y_pred.shape}, y_true {y_true.shape}")

        loss = nn.functional.mse_loss(y_pred, y_true)

        # manual backward and update
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                p -= lr * p.grad
        return float(loss.item())

    # ------------------------
    # Weight utilities
    # ------------------------
    def get_weights_numpy(self):
        """Return (w, b) as numpy arrays: w shape (D,), b scalar."""
        W = self.linear.weight.data.cpu().numpy().reshape(-1)
        b = float(self.linear.bias.data.cpu().numpy().reshape(-1)[0])
        return W.astype(np.float32), b

    def set_weights_numpy(self, w_np, b_scalar):
        """Set linear weights from numpy arrays (w_np shape D, b scalar)."""
        w_np = np.asarray(w_np, dtype=np.float32)
        with torch.no_grad():
            self.linear.weight.data[:] = torch.tensor(w_np.reshape(1, -1), dtype=torch.float32, device=self.device)
            self.linear.bias.data[:] = torch.tensor(float(b_scalar), dtype=torch.float32, device=self.device)

    def set_weights_from_ridge(self, w_np, b_scalar):
        """Alias for set_weights_numpy (semantic)."""
        self.set_weights_numpy(w_np, b_scalar)

    def soft_update_from_ridge(self, w_np, b_scalar, mix=0.2):
        """
        Soft-update readout weights towards ridge solution:
            new_w = mix * w_ridge + (1-mix) * w_old
        mix in [0,1], low mix => small transfer.
        """
        w_old, b_old = self.get_weights_numpy()
        if w_old.shape[0] != np.asarray(w_np).shape[0]:
            raise ValueError("Dimension mismatch between readout and provided ridge weights.")
        new_w = mix * np.asarray(w_np) + (1.0 - mix) * w_old
        new_b = mix * float(b_scalar) + (1.0 - mix) * float(b_old)
        self.set_weights_numpy(new_w, new_b)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        map_location = map_location or self.device
        self.load_state_dict(torch.load(path, map_location=map_location))


def ridge_regression_fit(H, Y, alpha=1e-3):
    """
    H: (N, D) numpy array
    Y: (N,) numpy array
    returns: w (D,) and b (scalar bias)
    """
    H = np.asarray(H, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    N, D = H.shape
    HtH = H.T @ H
    reg = alpha * np.eye(D)
    w = np.linalg.solve(HtH + reg, H.T @ Y)
    b = float(np.mean(Y) - np.mean(H, axis=0) @ w)
    return w.astype(np.float32), float(b)


# -----------------------------
# Quick test / demo
# -----------------------------
if __name__ == "__main__":
    print("Readout quick test...")
    D = 64
    import numpy as np

    # random features & targets
    H = np.random.randn(200, D).astype(np.float32)
    Y = (H.sum(axis=1) * 0.1 + 0.05 * np.random.randn(200)).astype(np.float32)

    # Fit ridge and inspect
    w_ridge, b_ridge = ridge_regression_fit(H, Y, alpha=1e-3)
    print("Ridge w shape:", w_ridge.shape, "bias:", b_ridge)

    # Torch readout online update demonstration
    readout = Readout(res_size=D)
    h_sample = torch.tensor(H[0], dtype=torch.float32)
    y_sample = float(Y[0])

    # compute loss before with matching shapes
    loss_before = nn.functional.mse_loss(readout(h_sample), torch.tensor([y_sample], dtype=torch.float32)).item()
    print(f"Loss before any updates: {loss_before:.6f}")

    # Demonstrate soft transfer from ridge into readout
    print("Applying soft transfer from ridge weights (mix=0.2) ...")
    readout.soft_update_from_ridge(w_ridge, b_ridge, mix=0.2)

    loss_after_soft = nn.functional.mse_loss(readout(h_sample), torch.tensor([y_sample], dtype=torch.float32)).item()
    print(f"Loss after soft update: {loss_after_soft:.6f}")

    # Demonstrate direct full set (hard replace)
    readout.set_weights_from_ridge(w_ridge, b_ridge)
    loss_after_set = nn.functional.mse_loss(readout(h_sample), torch.tensor([y_sample], dtype=torch.float32)).item()
    print(f"Loss after hard set to ridge weights: {loss_after_set:.6f}")

    # Show batch numpy predict
    preds = readout.predict_batch_numpy(H[:5])
    print("First 5 preds (numpy):", preds)

    # Save & load checkpoint
    ckpt = "models/test_readout.pth"
    readout.save(ckpt)
    print(f"Saved readout to {ckpt}")

    # reload into new module
    r2 = Readout(res_size=D)
    r2.load(ckpt)
    print("Reloaded checkpoint into a new Readout instance.")
    print("Test finished.")
