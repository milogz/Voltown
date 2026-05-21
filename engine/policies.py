"""
Policy Implementations — All Four Powell Classes
==================================================
Every policy maps state to decision:  X = Pi(R_t, W_t; theta)

All policies return the same decision dict:
  {'x_gb': float, 'x_sb': float, 'x_sd': float,
   'x_bd': float, 'x_gd': float}

Powell's taxonomy:
  PFA — direct parametric rule, no optimization
  CFA — single-period LP with modified costs
  VFA — greedy w.r.t. learned value function
  DLA — multi-period LP with forecast (rolling horizon)
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ═══════════════════════════════════════════════════════════════
# PFA — Policy Function Approximation
# ═══════════════════════════════════════════════════════════════

def policy_PFA_threshold(R_t, W_t, params):
    """PFA: threshold-based rule.

    If price is cheap and battery has room → charge from grid.
    If battery is full enough → discharge to demand.
    Otherwise → use solar, fill from grid.

    Parameters:
      theta1: price threshold (charge if p < theta1)
      theta2: battery cap for charging (charge if R < theta2)
      theta3: battery trigger for discharge (discharge if R > theta3)
    """
    p, d, s = W_t['price'], W_t['demand'], W_t['solar']
    K = params.get('K', 10.0)
    theta1 = params.get('theta1', 0.12)
    theta2 = params.get('theta2', 8.0)
    theta3 = params.get('theta3', 5.0)

    # Solar first: always use free energy
    x_sd = min(s, d)
    remaining_solar = s - x_sd
    remaining_demand = d - x_sd
    x_gb, x_sb, x_bd, x_gd = 0.0, 0.0, 0.0, 0.0

    if p < theta1 and R_t < theta2:
        charge_room = min(K - R_t, 5.0)
        x_sb = min(remaining_solar, charge_room)
        charge_room -= x_sb
        x_gb = charge_room
        x_gd = remaining_demand
    elif R_t > theta3:
        x_bd = min(R_t, remaining_demand)
        x_gd = max(0, remaining_demand - x_bd)
        x_sb = min(remaining_solar, K - R_t + x_bd)
    else:
        x_sb = min(remaining_solar, K - R_t)
        x_gd = remaining_demand

    return {'x_gb': x_gb, 'x_sb': x_sb, 'x_sd': x_sd,
            'x_bd': x_bd, 'x_gd': x_gd}


def policy_PFA_linear(R_t, W_t, params):
    """PFA: linear parametric rule (continuous signals).

    charge_signal   = max(0, θ₁ - θ₂·p + θ₃·(K - R))
    discharge_signal = max(0, θ₄·R - θ₅/p)

    This is literally a single-layer neural network with ReLU —
    connecting PFA to deep policy gradient methods.
    """
    p, d, s = W_t['price'], W_t['demand'], W_t['solar']
    K = params.get('K', 10.0)
    t1 = params.get('theta1', 1.0)
    t2 = params.get('theta2', 5.0)
    t3 = params.get('theta3', 0.3)
    t4 = params.get('theta4', 0.5)
    t5 = params.get('theta5', 0.01)

    charge_signal = max(0, t1 - t2 * p + t3 * (K - R_t))
    discharge_signal = max(0, t4 * R_t - t5 / max(p, 0.01))

    x_sd = min(s, d)
    remaining_solar = s - x_sd
    remaining_demand = d - x_sd

    if charge_signal > discharge_signal:
        x_sb = min(remaining_solar, K - R_t)
        x_gb = min(charge_signal, K - R_t - x_sb)
        x_bd = 0.0
        x_gd = remaining_demand
    else:
        x_sb = min(remaining_solar, K - R_t)
        x_gb = 0.0
        x_bd = min(discharge_signal, R_t, remaining_demand)
        x_gd = max(0, remaining_demand - x_bd)

    return {'x_gb': max(0, x_gb), 'x_sb': max(0, x_sb), 'x_sd': max(0, x_sd),
            'x_bd': max(0, x_bd), 'x_gd': max(0, x_gd)}


# ═══════════════════════════════════════════════════════════════
# CFA — Cost Function Approximation
# ═══════════════════════════════════════════════════════════════

def policy_CFA(R_t, W_t, params):
    """CFA: single-period LP with storage-value proxy.

    min  p·(x_gb + x_gd) - θ·R_{t+1}

    The proxy θ assigns value to stored energy, creating a "buffer"
    incentive without any forecast.  This is the engineering art
    of modifying the objective to implicitly value the future.
    """
    K = params.get('K', 10.0)
    theta = params.get('theta', 0.0)

    m = gp.Model('CFA')
    m.setParam('OutputFlag', 0)

    x_gb = m.addVar(name='x_gb')
    x_sb = m.addVar(name='x_sb')
    x_sd = m.addVar(name='x_sd')
    x_bd = m.addVar(name='x_bd', ub=R_t)
    x_gd = m.addVar(name='x_gd')

    R_next = R_t + x_gb + x_sb - x_bd

    m.addConstr(x_sb + x_sd <= W_t['solar'], 'solar_cap')
    m.addConstr(x_bd + x_gd + x_sd >= W_t['demand'], 'meet_demand')
    m.addConstr(R_next <= K, 'batt_ub')
    m.addConstr(R_next >= 0, 'batt_lb')

    m.setObjective(
        W_t['price'] * (x_gb + x_gd) - theta * R_next,
        GRB.MINIMIZE
    )
    m.optimize()

    return {k: m.getVarByName(k).X for k in
            ['x_gb', 'x_sb', 'x_sd', 'x_bd', 'x_gd']}


def policy_CFA_risk(R_t, W_t, params):
    """CFA + Risk: inflated buffer via price volatility.

    θ_risk = θ + κ · σ_price

    When prices are volatile, the policy assigns MORE value to stored
    energy → larger buffers → protection against worst-case spikes.
    """
    K = params.get('K', 10.0)
    theta = params.get('theta', 0.0)
    kappa = params.get('kappa', 1.0)
    sigma_price = params.get('sigma_price', 0.01)
    theta_risk = theta + kappa * sigma_price

    m = gp.Model('CFA_risk')
    m.setParam('OutputFlag', 0)

    x_gb = m.addVar(name='x_gb')
    x_sb = m.addVar(name='x_sb')
    x_sd = m.addVar(name='x_sd')
    x_bd = m.addVar(name='x_bd', ub=R_t)
    x_gd = m.addVar(name='x_gd')

    R_next = R_t + x_gb + x_sb - x_bd

    m.addConstr(x_sb + x_sd <= W_t['solar'], 'solar_cap')
    m.addConstr(x_bd + x_gd + x_sd >= W_t['demand'], 'meet_demand')
    m.addConstr(R_next <= K, 'batt_ub')
    m.addConstr(R_next >= 0, 'batt_lb')

    m.setObjective(
        W_t['price'] * (x_gb + x_gd) - theta_risk * R_next,
        GRB.MINIMIZE
    )
    m.optimize()

    return {k: m.getVarByName(k).X for k in
            ['x_gb', 'x_sb', 'x_sd', 'x_bd', 'x_gd']}


# ═══════════════════════════════════════════════════════════════
# VFA — Value Function Approximation
# ═══════════════════════════════════════════════════════════════

def policy_VFA_tabular(R_t, W_t, params):
    """VFA: look up optimal action from pre-computed V* table.

    Uses the discretized state space (R × price_regime × demand_regime)
    and the policy vector from Value Iteration.
    """
    from engine.vfa_training import (S_idx, feasible_actions,
                                     action_to_X, discretize_price,
                                     discretize_demand)
    K = params.get('K', 10.0)
    pi = params['policy']
    R_disc = int(np.clip(round(R_t), 0, K))
    pr = discretize_price(W_t['price'])
    dr = discretize_demand(W_t['demand'])
    s_idx = S_idx[(R_disc, pr, dr)]
    a = pi[s_idx]
    return action_to_X(a, R_disc, W_t, K)


def policy_VFA_qlearn(R_t, W_t, params):
    """VFA: greedy w.r.t. learned Q-table from Q-learning."""
    from engine.vfa_training import (S_idx, feasible_actions,
                                     action_to_X, discretize_price,
                                     discretize_demand)
    K = params.get('K', 10.0)
    Q = params['Q']
    R_disc = int(np.clip(round(R_t), 0, K))
    pr = discretize_price(W_t['price'])
    dr = discretize_demand(W_t['demand'])
    s_idx = S_idx[(R_disc, pr, dr)]
    feas = feasible_actions(R_disc, K)
    best_a = min(feas, key=lambda a: Q[s_idx, a])
    return action_to_X(best_a, R_disc, W_t, K)


def policy_VFA_adp(R_t, W_t, params):
    """VFA ADP: greedy w.r.t. linear V approximation."""
    from engine.vfa_training import (feasible_actions, action_to_X,
                                     discretize_price, phi)
    from engine.mdp import transition, period_cost
    K = params.get('K', 10.0)
    w = params['weights']
    gamma = params.get('gamma', 0.95)
    R_disc = int(np.clip(round(R_t), 0, K))
    pr = discretize_price(W_t['price'])

    best_a, best_q = 0, np.inf
    for a in feasible_actions(R_disc, K):
        X = action_to_X(a, R_disc, W_t, K)
        c = period_cost(X, W_t)
        R_next = int(np.clip(transition(R_disc, X, K), 0, K))
        q = c + gamma * phi(R_next, pr) @ w
        if q < best_q:
            best_q, best_a = q, a

    return action_to_X(best_a, R_disc, W_t, K)


# ═══════════════════════════════════════════════════════════════
# DLA — Direct Lookahead Approximation
# ═══════════════════════════════════════════════════════════════

def policy_DLA_det(R_t, W_t, W_forecast, params):
    """DLA Deterministic: rolling-horizon LP with point forecast.

    Optimizes over H periods using observed W_t (period 0) and
    forecasted W (periods 1..H-1).  Only implements period-0 decision.

    This is Model Predictive Control (MPC) — solve, execute first
    action, re-solve at next period with updated information.
    """
    K = params.get('K', 10.0)
    H = params.get('horizon', 4)
    fc_len = len(W_forecast['demand'])
    H = min(H, fc_len + 1)

    m = gp.Model('DLA_det')
    m.setParam('OutputFlag', 0)

    prices = np.concatenate([[W_t['price']], W_forecast['price'][:H - 1]])
    demands = np.concatenate([[W_t['demand']], W_forecast['demand'][:H - 1]])
    solars = np.concatenate([[W_t['solar']], W_forecast['solar'][:H - 1]])
    H = len(prices)
    periods = range(H)

    x_gb = {t: m.addVar(name=f'x_gb_{t}') for t in periods}
    x_sb = {t: m.addVar(name=f'x_sb_{t}') for t in periods}
    x_sd = {t: m.addVar(name=f'x_sd_{t}') for t in periods}
    x_bd = {t: m.addVar(name=f'x_bd_{t}') for t in periods}
    x_gd = {t: m.addVar(name=f'x_gd_{t}') for t in periods}
    B = {t: m.addVar(lb=0, ub=K, name=f'B_{t}') for t in periods}
    m.update()

    m.addConstr(B[0] == R_t + x_gb[0] + x_sb[0] - x_bd[0], 'trans_0')
    m.addConstr(x_bd[0] <= R_t, 'discharge_0')
    for t in range(1, H):
        m.addConstr(B[t] == B[t - 1] + x_gb[t] + x_sb[t] - x_bd[t])
        m.addConstr(x_bd[t] <= B[t - 1])
    for t in periods:
        m.addConstr(x_sb[t] + x_sd[t] <= solars[t])
        m.addConstr(x_bd[t] + x_gd[t] + x_sd[t] >= demands[t])

    m.setObjective(
        gp.quicksum(prices[t] * (x_gb[t] + x_gd[t]) for t in periods),
        GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        # Fallback: just meet demand from grid
        d, s = W_t['demand'], W_t['solar']
        x_sd_v = min(s, d)
        return {'x_gb': 0, 'x_sb': 0, 'x_sd': x_sd_v,
                'x_bd': 0, 'x_gd': max(0, d - x_sd_v)}

    return {k: m.getVarByName(f'{k}_0').X for k in
            ['x_gb', 'x_sb', 'x_sd', 'x_bd', 'x_gd']}


def policy_DLA_stoch(R_t, W_t, W_scenarios, params):
    """DLA Stochastic: rolling-horizon with scenario fan.

    Period-0 decisions are forced to coincide across all scenarios
    via NON-ANTICIPATIVITY constraints — this is exactly two-stage SP
    from Module 2, embedded in a rolling loop.
    """
    K = params.get('K', 10.0)
    H = params.get('horizon', 4)
    num_sc = W_scenarios['demand'].shape[0]
    fc_len = W_scenarios['demand'].shape[1]
    H = min(H, fc_len + 1)
    prob = 1.0 / num_sc

    m = gp.Model('DLA_stoch')
    m.setParam('OutputFlag', 0)
    periods = range(H)
    scenarios = range(num_sc)

    x = {}
    for w in scenarios:
        for t in periods:
            for k in ['x_gb', 'x_sb', 'x_sd', 'x_bd', 'x_gd']:
                x[k, t, w] = m.addVar(name=f'{k}_{t}_{w}')

    B = {}
    for w in scenarios:
        for t in periods:
            B[t, w] = m.addVar(lb=0, ub=K, name=f'B_{t}_{w}')
    m.update()

    # Non-anticipativity: period-0 decisions coincide
    for w in range(1, num_sc):
        for k in ['x_gb', 'x_sb', 'x_sd', 'x_bd', 'x_gd']:
            m.addConstr(x[k, 0, w] == x[k, 0, 0], f'NA_{k}_{w}')

    for w in scenarios:
        prices = np.concatenate([[W_t['price']], W_scenarios['price'][w, :H - 1]])
        demands = np.concatenate([[W_t['demand']], W_scenarios['demand'][w, :H - 1]])
        solars = np.concatenate([[W_t['solar']], W_scenarios['solar'][w, :H - 1]])

        m.addConstr(B[0, w] == R_t + x['x_gb', 0, w] + x['x_sb', 0, w]
                    - x['x_bd', 0, w])
        m.addConstr(x['x_bd', 0, w] <= R_t)
        m.addConstr(x['x_sb', 0, w] + x['x_sd', 0, w] <= W_t['solar'])
        m.addConstr(x['x_bd', 0, w] + x['x_gd', 0, w] + x['x_sd', 0, w]
                    >= W_t['demand'])

        for t in range(1, H):
            xg, xs, xd = x['x_gb', t, w], x['x_sb', t, w], x['x_sd', t, w]
            xb, xG = x['x_bd', t, w], x['x_gd', t, w]
            m.addConstr(B[t, w] == B[t - 1, w] + xg + xs - xb)
            m.addConstr(xb <= B[t - 1, w])
            m.addConstr(xs + xd <= solars[t])
            m.addConstr(xb + xG + xd >= demands[t])

    m.setObjective(
        gp.quicksum(
            prob * np.concatenate(
                [[W_t['price']], W_scenarios['price'][w, :H - 1]])[t]
            * (x['x_gb', t, w] + x['x_gd', t, w])
            for w in scenarios for t in periods),
        GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        d, s = W_t['demand'], W_t['solar']
        x_sd_v = min(s, d)
        return {'x_gb': 0, 'x_sb': 0, 'x_sd': x_sd_v,
                'x_bd': 0, 'x_gd': max(0, d - x_sd_v)}

    return {k: x[k, 0, 0].X for k in
            ['x_gb', 'x_sb', 'x_sd', 'x_bd', 'x_gd']}


def policy_DLA_stoch_cvar(R_t, W_t, W_scenarios, params):
    """DLA + Risk: Mean-CVaR weighted objective.

    λ·E[cost] + (1-λ)·CVaR_α[cost]

    Uses the Rockafellar-Uryasev linearization from Module 2.
    """
    K = params.get('K', 10.0)
    H = params.get('horizon', 4)
    lam = params.get('lambda', 0.5)
    alpha = params.get('alpha', 0.05)
    num_sc = W_scenarios['demand'].shape[0]
    fc_len = W_scenarios['demand'].shape[1]
    H = min(H, fc_len + 1)
    prob = 1.0 / num_sc

    m = gp.Model('DLA_cvar')
    m.setParam('OutputFlag', 0)
    periods = range(H)
    scenarios = range(num_sc)

    x = {}
    for w in scenarios:
        for t in periods:
            for k in ['x_gb', 'x_sb', 'x_sd', 'x_bd', 'x_gd']:
                x[k, t, w] = m.addVar(name=f'{k}_{t}_{w}')

    B = {}
    for w in scenarios:
        for t in periods:
            B[t, w] = m.addVar(lb=0, ub=K, name=f'B_{t}_{w}')

    eta = m.addVar(lb=-GRB.INFINITY, name='eta')
    z = {w: m.addVar(name=f'z_{w}') for w in scenarios}
    sc_cost = {w: m.addVar(lb=-GRB.INFINITY, name=f'sc_cost_{w}')
               for w in scenarios}
    m.update()

    # Non-anticipativity
    for w in range(1, num_sc):
        for k in ['x_gb', 'x_sb', 'x_sd', 'x_bd', 'x_gd']:
            m.addConstr(x[k, 0, w] == x[k, 0, 0], f'NA_{k}_{w}')

    for w in scenarios:
        prices = np.concatenate([[W_t['price']], W_scenarios['price'][w, :H - 1]])
        demands = np.concatenate([[W_t['demand']], W_scenarios['demand'][w, :H - 1]])
        solars = np.concatenate([[W_t['solar']], W_scenarios['solar'][w, :H - 1]])

        m.addConstr(B[0, w] == R_t + x['x_gb', 0, w] + x['x_sb', 0, w]
                    - x['x_bd', 0, w])
        m.addConstr(x['x_bd', 0, w] <= R_t)
        m.addConstr(x['x_sb', 0, w] + x['x_sd', 0, w] <= W_t['solar'])
        m.addConstr(x['x_bd', 0, w] + x['x_gd', 0, w] + x['x_sd', 0, w]
                    >= W_t['demand'])

        for t in range(1, H):
            xg, xs, xd = x['x_gb', t, w], x['x_sb', t, w], x['x_sd', t, w]
            xb, xG = x['x_bd', t, w], x['x_gd', t, w]
            m.addConstr(B[t, w] == B[t - 1, w] + xg + xs - xb)
            m.addConstr(xb <= B[t - 1, w])
            m.addConstr(xs + xd <= solars[t])
            m.addConstr(xb + xG + xd >= demands[t])

        m.addConstr(sc_cost[w] ==
                    gp.quicksum(prices[t] * (x['x_gb', t, w] + x['x_gd', t, w])
                                for t in periods))
        m.addConstr(z[w] >= sc_cost[w] - eta)

    exp_cost = gp.quicksum(prob * sc_cost[w] for w in scenarios)
    cvar = eta + (1.0 / alpha) * gp.quicksum(prob * z[w] for w in scenarios)
    m.setObjective(lam * exp_cost + (1 - lam) * cvar, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        d, s = W_t['demand'], W_t['solar']
        x_sd_v = min(s, d)
        return {'x_gb': 0, 'x_sb': 0, 'x_sd': x_sd_v,
                'x_bd': 0, 'x_gd': max(0, d - x_sd_v)}

    return {k: x[k, 0, 0].X for k in
            ['x_gb', 'x_sb', 'x_sd', 'x_bd', 'x_gd']}


# ═══════════════════════════════════════════════════════════════
# Policy Registry — for dashboard UI
# ═══════════════════════════════════════════════════════════════

POLICY_REGISTRY = {
    'PFA Threshold': {
        'fn': policy_PFA_threshold,
        'class': 'PFA',
        'color': '#e74c3c',
        'needs_forecast': False,
        'params': {
            'theta1': {'label': 'Price threshold ($/kWh)', 'min': 0.06, 'max': 0.20, 'default': 0.12, 'step': 0.01},
            'theta2': {'label': 'Charge cap (kWh)', 'min': 3.0, 'max': 10.0, 'default': 8.0, 'step': 0.5},
            'theta3': {'label': 'Discharge trigger (kWh)', 'min': 2.0, 'max': 9.0, 'default': 5.0, 'step': 0.5},
        }
    },
    'PFA Linear': {
        'fn': policy_PFA_linear,
        'class': 'PFA',
        'color': '#e67e22',
        'needs_forecast': False,
        'params': {
            'theta1': {'label': 'Charge bias', 'min': 0.0, 'max': 3.0, 'default': 1.0, 'step': 0.1},
            'theta2': {'label': 'Price weight', 'min': 0.0, 'max': 10.0, 'default': 5.0, 'step': 0.5},
            'theta3': {'label': 'Capacity weight', 'min': 0.0, 'max': 1.0, 'default': 0.3, 'step': 0.05},
            'theta4': {'label': 'Discharge weight', 'min': 0.0, 'max': 2.0, 'default': 0.5, 'step': 0.1},
            'theta5': {'label': 'Price retention', 'min': 0.0, 'max': 0.1, 'default': 0.01, 'step': 0.005},
        }
    },
    'CFA Buffer': {
        'fn': policy_CFA,
        'class': 'CFA',
        'color': '#3498db',
        'needs_forecast': False,
        'params': {
            'theta': {'label': 'Storage value θ ($/kWh)', 'min': 0.0, 'max': 0.25, 'default': 0.10, 'step': 0.01},
        }
    },
    'CFA Risk-Averse': {
        'fn': policy_CFA_risk,
        'class': 'CFA',
        'color': '#2980b9',
        'needs_forecast': False,
        'params': {
            'theta': {'label': 'Base θ', 'min': 0.0, 'max': 0.20, 'default': 0.10, 'step': 0.01},
            'kappa': {'label': 'Risk aversion κ', 'min': 0.0, 'max': 5.0, 'default': 2.0, 'step': 0.5},
        }
    },
    'VFA Tabular (VI)': {
        'fn': policy_VFA_tabular,
        'class': 'VFA',
        'color': '#2ecc71',
        'needs_forecast': False,
        'needs_training': 'vi',
        'params': {}
    },
    'VFA Q-Learning': {
        'fn': policy_VFA_qlearn,
        'class': 'VFA',
        'color': '#1abc9c',
        'needs_forecast': False,
        'needs_training': 'ql',
        'params': {}
    },
    'DLA Deterministic': {
        'fn': policy_DLA_det,
        'class': 'DLA',
        'color': '#9b59b6',
        'needs_forecast': True,
        'params': {
            'horizon': {'label': 'Horizon H', 'min': 2, 'max': 12, 'default': 6, 'step': 1},
        }
    },
    'DLA Stochastic': {
        'fn': policy_DLA_stoch,
        'class': 'DLA',
        'color': '#8e44ad',
        'needs_forecast': True,
        'params': {
            'horizon': {'label': 'Horizon H', 'min': 2, 'max': 12, 'default': 4, 'step': 1},
            'num_scenarios': {'label': 'Scenarios', 'min': 3, 'max': 20, 'default': 5, 'step': 1},
        }
    },
    'DLA CVaR': {
        'fn': policy_DLA_stoch_cvar,
        'class': 'DLA',
        'color': '#c0392b',
        'needs_forecast': True,
        'params': {
            'horizon': {'label': 'Horizon H', 'min': 2, 'max': 12, 'default': 4, 'step': 1},
            'num_scenarios': {'label': 'Scenarios', 'min': 3, 'max': 20, 'default': 5, 'step': 1},
            'lambda': {'label': 'λ (risk blend)', 'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.1},
            'alpha': {'label': 'α (CVaR level)', 'min': 0.01, 'max': 0.50, 'default': 0.05, 'step': 0.01},
        }
    },
}

# Class colors for consistent visualization
CLASS_COLORS = {
    'PFA': '#e74c3c',
    'CFA': '#3498db',
    'VFA': '#2ecc71',
    'DLA': '#9b59b6',
}
