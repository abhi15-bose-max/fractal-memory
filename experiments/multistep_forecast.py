"""
Multi-step forecasting: compare 1-step, 5-step, 10-step ahead forecasting
using same fractal pipeline (fast online + consolidation).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from src.data import gen_drifting_sine
from src.fractal_model import FractalModel
from src.utils import mse, seed_everything

OUT="results/plots"
os.makedirs(OUT, exist_ok=True)
seed_everything(123)

T=1400
x = gen_drifting_sine(T=T, seed=123)

def run_k_forecast(k):
    m = FractalModel(res_size=200, reservoir_kwargs={"seed":42}, buffer_maxlen=800, online_lr=1e-4, use_ae=False)
    preds_fast = np.zeros(T, dtype=np.float32)
    preds_slow = np.zeros(T, dtype=np.float32)
    for t in range(T - k - 1):
        y_fast, h = m.step(x[t])
        preds_fast[t] = y_fast
        m.add_to_buffer(h, x[t+k])  # target is k-step ahead
        # online update toward k-step target (still using immediate next-step as proxy, keep it simple)
        import torch
        m.readout.online_sgd_step(torch.tensor(h, dtype=torch.float32), float(x[t+k]), lr=m.online_lr)
        if (t+1) % 300 == 0 and m.buffer.size() >= 50:
            m.maybe_consolidate(min_samples=50, alpha=1e-3)
        sp = m.slow_predict(h)
        preds_slow[t] = sp if sp is not None else preds_fast[t]
    return preds_fast, preds_slow

for k in [1,5,10]:
    pf, ps = run_k_forecast(k)
    # evaluate only on valid indices
    valid = slice(0, len(pf))
    print(f"k={k}  fast MSE={mse(x[:len(pf)], pf):.6f}  slow MSE={mse(x[:len(ps)], ps):.6f}")
    # save a plot for each k
    plt.figure(figsize=(10,3))
    t0 = 350
    plt.plot(x[t0:t0+200], label="signal")
    plt.plot(pf[t0:t0+200], label=f"fast k={k}")
    plt.plot(ps[t0:t0+200], label=f"slow k={k}")
    plt.legend()
    plt.title(f"Multi-step forecast k={k}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"multistep_k{k}.png"), dpi=150)
    plt.show()
