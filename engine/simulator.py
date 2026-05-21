"""
Simulator — Episode Runner and Tournament Engine
==================================================
Connects policies to the MDP environment and runs episodes.

This is the "main loop" of any sequential decision problem:
  for t in range(T):
      observe state → decide → observe exogenous → transition → record
"""

import numpy as np
import pandas as pd
from engine.mdp import transition, period_cost
from engine.forecast import forecast_naive


def run_episode(policy_fn, W, params, K=10.0, R_init=2.0,
                needs_forecast=False):
    """Run one complete episode of T sequential decisions.

    This is the core simulation loop — Powell's "online" execution.

    Returns dict with:
      costs:     array of per-period costs
      battery:   array of battery levels (length T+1)
      decisions: list of decision dicts
      total_cost: scalar sum
    """
    T = len(W['demand'])
    R = R_init
    costs, battery, decisions = [], [R], []

    for t in range(T):
        W_t = {k: v[t] for k, v in W.items()}

        if needs_forecast:
            remaining = T - t - 1
            if remaining > 0:
                W_hist = {k: v[:max(1, t)] for k, v in W.items()}
                H = min(params.get('horizon', 4), remaining)
                num_paths = params.get('num_scenarios', 5)
                np.random.seed(hash((t, params.get('_seed', 0))) % 2**31)
                W_fc = forecast_naive(W_hist, H, num_paths=num_paths)

                # DLA det uses a single path; DLA stoch uses all paths
                if num_paths == 1 or 'det' in params.get('_variant', 'det'):
                    W_fc_single = {k: v[0] for k, v in W_fc.items()}
                    X = policy_fn(R, W_t, W_fc_single, {**params, 'K': K})
                else:
                    X = policy_fn(R, W_t, W_fc, {**params, 'K': K})
            else:
                # Last period: just meet demand
                d, s = W_t['demand'], W_t['solar']
                x_sd = min(s, d)
                X = {'x_gb': 0, 'x_sb': 0, 'x_sd': x_sd,
                     'x_bd': 0, 'x_gd': max(0, d - x_sd)}
        else:
            X = policy_fn(R, W_t, {**params, 'K': K})

        c = period_cost(X, W_t)
        R = transition(R, X, K)
        costs.append(c)
        battery.append(R)
        decisions.append(X)

    return {
        'costs': np.array(costs),
        'battery': np.array(battery),
        'decisions': decisions,
        'total_cost': sum(costs),
    }


def run_tournament(policy_configs, generate_W_fn, world_params,
                   n_episodes=30, seed_start=42):
    """Run a multi-episode tournament comparing several policies.

    policy_configs: dict of {name: {'fn', 'params', 'needs_forecast'}}
    Returns a DataFrame with columns = policy names, rows = episodes.
    """
    seeds = [seed_start + 100 * i for i in range(n_episodes)]
    results = {name: [] for name in policy_configs}

    for seed in seeds:
        W = generate_W_fn(world_params, seed=seed)
        for name, cfg in policy_configs.items():
            params = {**cfg.get('params', {}), '_seed': seed}
            if cfg.get('needs_forecast'):
                params['_variant'] = 'stoch' if cfg.get('stochastic') else 'det'
            r = run_episode(
                cfg['fn'], W, params,
                K=world_params.K,
                R_init=world_params.R_init,
                needs_forecast=cfg.get('needs_forecast', False)
            )
            results[name].append(r['total_cost'])

    return pd.DataFrame(results)


def compute_tournament_stats(df):
    """Compute summary statistics from tournament DataFrame.

    Returns a DataFrame with mean, std, min, max, CVaR90 per policy.
    """
    stats = {}
    for col in df.columns:
        arr = df[col].values
        sorted_arr = np.sort(arr)
        n_tail = max(1, int(len(arr) * 0.1))  # worst 10%
        cvar90 = np.mean(sorted_arr[-n_tail:])
        stats[col] = {
            'Mean': np.mean(arr),
            'Std': np.std(arr),
            'Best': np.min(arr),
            'Worst': np.max(arr),
            'CVaR₉₀': cvar90,
        }
    return pd.DataFrame(stats).T
