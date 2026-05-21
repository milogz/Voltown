"""
MDP Core — Transition and Cost Functions
==========================================
The physical "rules" of the Voltown battery system.

Powell's elements:
  Transition:   S_{t+1} = S^M(S_t, x_t, W_{t+1})
  Contribution: C_t = C(S_t, x_t, W_t)
"""

import numpy as np


def transition(R: float, X: dict, K: float = 10.0) -> float:
    """Battery state transition.

    R_{t+1} = clip(R_t + x_gb + x_sb - x_bd, 0, K)

    The clip enforces physical battery bounds: you can't store
    more than capacity or discharge below zero.
    """
    R_next = R + X['x_gb'] + X['x_sb'] - X['x_bd']
    return float(np.clip(R_next, 0, K))


def period_cost(X: dict, W_t: dict) -> float:
    """Grid energy cost at period t.

    C_t = p_t * (x_gb + x_gd)

    We pay grid price for every kWh drawn from the grid,
    whether it goes to the battery (x_gb) or directly to demand (x_gd).
    Solar is free.
    """
    return W_t['price'] * (X['x_gb'] + X['x_gd'])
