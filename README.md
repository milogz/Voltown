# 🔋 Voltown Decision Lab

Interactive dashboard for exploring **sequential decision-making under uncertainty** through an energy storage problem.

Built as teaching material for *Optimización bajo Incertidumbre* — Universidad de los Andes.

**Live:** [voltown.decision-lab.co](https://voltown.decision-lab.co)

---

## The Problem

Voltown is a community with solar panels, a shared battery, and grid connection. Each period, an operator decides how to route energy (5 decision variables) to minimize electricity cost under uncertain prices, demand, and solar availability.

## Pages

| Page | Content |
|------|---------|
| **📐 Formulation** | MDP elements, constraints, 4 policy classes (PFA, CFA, VFA, DLA), risk formulations |
| **🔬 Laboratory** | World exploration, single-policy evaluation, multi-policy tournament |
| **🏗️ Architecture** | Code structure, interfaces, replication guide |

## Engine

Six modules implementing Powell's framework:

```
world.py → mdp.py → forecast.py → policies.py → simulator.py → vfa_training.py
```

**9 policies** across 4 classes: PFA (Threshold, Linear), CFA (Buffer, Risk-Averse), VFA (VI, Q-Learning, ADP), DLA (Deterministic, Stochastic, CVaR).

## Run Locally

```bash
pip install -r requirements.txt
streamlit run Inicio.py
```

Requires [Gurobi](https://www.gurobi.com/) for CFA and DLA policies (free academic license available).

## Deploy

Configured for [Render](https://render.com/) via `render.yaml`. Connect the repo and deploy.

---

*Based on Powell (2022) — Reinforcement Learning and Stochastic Optimization.*
