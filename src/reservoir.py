# reservoir (fast memory)
"""
src/reservoir.py
Echo-State style reservoir (fast memory) implementation.

Class: Reservoir
- step(u): advance one time-step with input u (scalar or 1D array)
- reset_state(seed=None): reset internal state to zeros (seed optional for reproducibility)
- get_state(): returns current hidden state (torch tensor)

Test at bottom demonstrates stepping through a drifting sine.
"""

import numpy as np
import torch


class Reservoir:
    def __init__(self, in_size=1, res_size=300, spectral_radius=0.95, leak=0.25, seed=0, device="cpu"):
        self.in_size = in_size
        self.res_size = res_size
        self.leak = float(leak)
        self.device = device

        rng = np.random.RandomState(seed)
        Win = rng.uniform(-0.5, 0.5, (res_size, in_size)).astype(np.float32)
        W = rng.randn(res_size, res_size).astype(np.float32)

        # scale to spectral radius
        try:
            eigvals = np.linalg.eigvals(W)
            sr = max(abs(eigvals))
            if sr == 0:
                sr = 1.0
        except Exception:
            sr = np.max(np.abs(W).sum(axis=1)) + 1e-6

        W *= (spectral_radius / (sr + 1e-9))

        self.W_in = torch.from_numpy(Win).to(device)
        self.W = torch.from_numpy(W).to(device)
        self.state = torch.zeros(res_size, dtype=torch.float32, device=device)
        self.act = torch.tanh

    def step(self, u):
        """
        Take input u (scalar or array-like of length in_size), update state, and return state (torch tensor).
        """
        if np.isscalar(u):
            u_vec = torch.tensor([u], dtype=torch.float32, device=self.device)
        else:
            u_vec = torch.tensor(u, dtype=torch.float32, device=self.device).view(-1)

        pre = self.W @ self.state + self.W_in @ u_vec
        new = (1.0 - self.leak) * self.state + self.leak * self.act(pre)
        self.state = new
        return self.state.clone()

    def reset_state(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.state = torch.zeros(self.res_size, dtype=torch.float32, device=self.device)

    def get_state(self):
        return self.state.clone()


if __name__ == "__main__":
    # Quick runtime test
    print("Reservoir quick test...")
    from src.data import gen_drifting_sine  # relative import expects package mode; adjust if running direct

    x = gen_drifting_sine(T=500, seed=2)
    r = Reservoir(in_size=1, res_size=64, spectral_radius=0.9, leak=0.2, seed=42)
    states = []
    for t in range(200):
        h = r.step(x[t])
        states.append(h.numpy())
    states = np.stack(states, axis=0)
    print("Reservoir states shape (200, res_size):", states.shape)
    print("Test finished.")
