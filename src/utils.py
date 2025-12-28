# misc utilities
"""
src/utils.py
Small utilities: seeding, metrics, simple plotting helper.
"""

import numpy as np
import random
import torch


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mse(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean((a - b) ** 2))


if __name__ == "__main__":
    print("Utils quick test...")
    seed_everything(123)
    a = np.arange(10, dtype=np.float32)
    b = a + 0.5
    print("MSE test (expected 0.25):", mse(a, b))
    print("Test finished.")
