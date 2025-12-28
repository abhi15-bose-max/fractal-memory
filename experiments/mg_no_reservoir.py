"""
experiments/mg_no_reservoir.py
Ablation: REMOVE RESERVOIR.
We use raw input x[t] as the “state”.
"""
from src.mackey_glass import gen_mackey_glass_sequence
from src.fractal_model import FractalModel
from src.utils import mse
import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import matplotlib.pyplot as plt
from src.mackey_glass import gen_mackey_glass_sequence
from src.utils import mse
from src.readout import Readout, ridge_regression_fit


class NoReservoirModel:
    def __init__(self, online_lr=1e-4, buffer_maxlen=1500):
        self.readout = Readout(res_size=1)  # scalar input
        self.buffer = []
        self.buffer_maxlen = buffer_maxlen
        self.slow_w = None
        self.slow_b = None
        self.online_lr = online_lr

    def step(self, x_t):
        import torch
        xt = torch.tensor([x_t], dtype=torch.float32)
        y_fast = float(self.readout(xt).item())
        return y_fast, np.array([x_t], dtype=np.float32)

    def add(self, h, y):
        if len(self.buffer) >= self.buffer_maxlen:
            self.buffer.pop(0)
        self.buffer.append((h, y))

    def consolidate(self):
        if len(self.buffer) < 40:
            return
        H = np.stack([b[0] for b in self.buffer], axis=0)
        Y = np.array([b[1] for b in self.buffer])
        self.slow_w, self.slow_b = ridge_regression_fit(H, Y)

    def slow_predict(self, h):
        if self.slow_w is None:
            return None
        return float(h.dot(self.slow_w) + self.slow_b)


if __name__ == "__main__":
    print("=== MG Ablation: NO RESERVOIR ===")

    x = gen_mackey_glass_sequence(T=3000, seed=12)
    model = NoReservoirModel()

    preds_fast = np.zeros_like(x)
    preds_slow = np.zeros_like(x)

    import torch
    for t in range(len(x) - 1):
        y_fast, h = model.step(x[t])
        preds_fast[t] = y_fast

        model.add(h, x[t + 1])

        # online SGD
        ht = torch.tensor(h, dtype=torch.float32)
        yt = torch.tensor([x[t + 1]], dtype=torch.float32)
        model.readout.online_sgd_step(ht, yt, lr=model.online_lr)

        if t % 400 == 0:
            model.consolidate()

        sp = model.slow_predict(h)
        preds_slow[t] = sp if sp is not None else y_fast

    print("Fast MSE:", mse(preds_fast, x))
    print("Slow MSE:", mse(preds_slow, x))

    plt.figure(figsize=(10, 4))
    plt.plot(x[:800], label="MG true")
    plt.plot(preds_fast[:800], label="fast")
    plt.plot(preds_slow[:800], label="slow")
    plt.title("MG: No Reservoir Ablation")
    plt.legend()
    plt.tight_layout()
    plt.show()
