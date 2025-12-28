"""
No episodic buffer:
- Use FractalModel but disable buffer additions and consolidation.
- Fast readout still does online SGD every step.
- Compare to baseline full model (with buffer+consolidation).
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

# run without buffer (simulate by setting buffer_maxlen=1 and skipping add)
def run_no_buffer():
    m = FractalModel(res_size=200, reservoir_kwargs={"seed":42}, buffer_maxlen=1, online_lr=1e-4, use_ae=False)
    pf = np.zeros(T, dtype=np.float32)
    ps = np.zeros(T, dtype=np.float32)
    for t in range(T-1):
        y_fast, h = m.step(x[t])
        pf[t] = y_fast
        # DO NOT add to buffer
        # online update only:
        import torch
        m.readout.online_sgd_step(torch.tensor(h, dtype=torch.float32), float(x[t+1]), lr=m.online_lr)
        sp = m.slow_predict(h)
        ps[t] = sp if sp is not None else pf[t]
    pf[-1]=pf[-2]; ps[-1]=ps[-2]
    return pf, ps

# run normal model for comparison
def run_with_buffer():
    m = FractalModel(res_size=200, reservoir_kwargs={"seed":42}, buffer_maxlen=800, online_lr=1e-4, use_ae=False)
    out = m.run_sequence(x, consolidate_every=300, min_consolidate_samples=50)
    return out["preds_fast"], out["preds_slow"]

pf_nb, ps_nb = run_no_buffer()
pf_b, ps_b = run_with_buffer()

print("MSEs:")
print("No-buffer fast:", mse(x, pf_nb))
print("With-buffer fast:", mse(x, pf_b))
print("With-buffer slow:", mse(x, ps_b))

# plot rolling errors
plt.figure(figsize=(10,5))
plt.plot(x, label="signal", alpha=0.4)
plt.plot(pf_nb, label="fast no buffer")
plt.plot(pf_b, label="fast with buffer")
plt.plot(ps_b, label="slow with buffer")
plt.legend()
plt.title("No episodic buffer ablation")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "no_episodic_buffer.png"), dpi=150)
plt.show()
