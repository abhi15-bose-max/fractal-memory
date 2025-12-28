"""
No fast memory: disable online SGD but allow buffer collection + periodic consolidation.
- The readout is not updated online (online_lr=0).
- Consolidation still happens every K steps and produces slow predictor.
- Evaluate slow predictor performance.
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

T=1200
x = gen_drifting_sine(T=T, seed=123)

m = FractalModel(res_size=200, reservoir_kwargs={"seed":42}, buffer_maxlen=800, online_lr=0.0, use_ae=False)
pf = np.zeros(T, dtype=np.float32)
ps = np.zeros(T, dtype=np.float32)

for t in range(T-1):
    y_fast, h = m.step(x[t])
    pf[t] = y_fast  # but fast isn't being updated
    m.add_to_buffer(h, x[t+1])
    # no online sgd (online_lr = 0)
    if (t+1) % 300 == 0 and m.buffer.size() >= 50:
        m.maybe_consolidate(min_samples=50, alpha=1e-3)
    sp = m.slow_predict(h)
    ps[t] = sp if sp is not None else pf[t]
pf[-1]=pf[-2]; ps[-1]=ps[-2]

print("MSE fast (no updates):", mse(x, pf))
print("MSE slow (only slow):", mse(x, ps))

plt.figure(figsize=(10,4))
plt.plot(x, label="signal", alpha=0.4)
plt.plot(pf, label="fast (no updates)")
plt.plot(ps, label="slow (consolidated)")
plt.legend()
plt.title("No fast memory (only slow consolidated predictor)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "no_fast_only_slow.png"), dpi=150)
plt.show()
