# Fractal Memory AI

# Fractal Memory: Multi-Timescale Memory for Non-Stationary Time-Series Prediction

Fractal Memory is a research prototype that implements a **three-timescale memory architecture** for time-series prediction under non-stationary and drifting dynamics.  
The model combines ideas from **reservoir computing**, **episodic replay**, and **slow consolidation**, inspired by Complementary Learning Systems (CLS).

The goal of this project is not to outperform all models on stationary benchmarks, but to **study stability–plasticity trade-offs** and memory dynamics under **fast, slow, and multi-scale drift**.

---

##  Core Idea

A *single Fractal Memory layer* consists of three interacting timescales:

1. **Fast memory (Reservoir dynamics)**  
   - A fixed recurrent reservoir (ESN-style)
   - Captures short-term temporal context

2. **Medium memory (Episodic buffer)**  
   - Stores recent reservoir states (optionally compressed)
   - Acts as a replay buffer

3. **Slow memory (Consolidation)**  
   - Periodic ridge-regression consolidation from episodic samples
   - Produces stable long-term predictors

These components together form a **single-layer, multi-timescale memory system**.

---





---

##  Experiments Included

- **Baseline comparison** (Fractal vs ESN vs LSTM vs GRU vs MLP)
- **Ablation studies**
  - No consolidation
  - No episodic buffer
  - Reservoir-only
  - Slow-only
- **Multi-step forecasting** (k = 1 … 30)
- **Drift sensitivity analysis**
  - Fast drift
  - Slow drift
  - Medium drift
  - Multi-scale drift
- **Chaotic systems**
  - Mackey–Glass
- **Synthetic non-stationary generators**
  - AM/FM, piecewise, heteroskedastic, random-walk drift

---

##  Metrics

The following metrics are computed across experiments:

- MSE / RMSE
- MAE
- SMAPE
- Dynamic Time Warping (DTW)
- Temporal lag error

The emphasis is on **relative behavior under drift**, not absolute performance on stationary data.

---

##  Important Notes

- This codebase is **frozen** for reproducibility.
- The model is intentionally **single-layer with three timescales**.
- Results show strengths under **fast and slow drift**, and weaknesses under stationary conditions.
- This behavior is expected and discussed as an inductive bias, not a flaw.

---

##  Future Work

Planned extensions include:
- Multi-layer Fractal Memory (stacked memory hierarchies)
- Adaptive consolidation schedules
- Theoretical analysis of stability–plasticity trade-offs
- Applications to control and reinforcement learning

---

## Reproducibility

All experiments were run on CPU using fixed random seeds.  
No GPU is required.

---

## License

This project is released under the MIT License (see LICENSE file).

---


