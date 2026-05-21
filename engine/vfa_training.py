"""
VFA Training — Value Iteration, Q-Learning, ADP
=================================================
Training algorithms for Value Function Approximation policies.

These are "offline" algorithms that pre-compute value functions
or Q-tables.  The resulting artifacts (V*, Q*, weights) are then
used by the corresponding policy functions in policies.py.

Discretization helpers are also here, since they define the
state/action space used by all VFA methods.
"""

import numpy as np
from engine.mdp import transition, period_cost


# ═══════════════════════════════════════════════════════════════
# State Space Discretization
# ═══════════════════════════════════════════════════════════════

PRICE_BINS = [0.0, 0.10, 0.14, 1.0]
DEMAND_BINS = [0.0, 2.5, 3.5, 10.0]

# Discrete actions: (charge_from_grid, discharge_to_demand)
ACTIONS = [
    (0, 0),   # hold
    (1, 0),   # charge +1
    (3, 0),   # charge +3
    (0, 1),   # discharge -1
    (0, 3),   # discharge -3
]
ACTION_LABELS = ['Hold', 'Charge+1', 'Charge+3', 'Disch-1', 'Disch-3']
N_ACTIONS = len(ACTIONS)


def discretize_price(p):
    """Map continuous price to regime index {0=low, 1=med, 2=high}."""
    if p < PRICE_BINS[1]: return 0
    if p < PRICE_BINS[2]: return 1
    return 2


def discretize_demand(d):
    """Map continuous demand to regime index {0=low, 1=med, 2=high}."""
    if d < DEMAND_BINS[1]: return 0
    if d < DEMAND_BINS[2]: return 1
    return 2


def build_state_space(K=10):
    """Build the enumerated state space (R, price_regime, demand_regime).

    Returns (STATES list, S_idx dict).
    """
    battery_levels = list(range(int(K) + 1))
    STATES = [(r, pr, dr) for r in battery_levels
              for pr in range(3) for dr in range(3)]
    S_idx = {s: i for i, s in enumerate(STATES)}
    return STATES, S_idx


# Pre-build default state space
STATES, S_idx = build_state_space(10)


def feasible_actions(R, K=10):
    """Return indices of feasible actions given battery level R."""
    feasible = []
    for i, (charge, discharge) in enumerate(ACTIONS):
        R_next = R + charge - discharge
        if 0 <= R_next <= K:
            feasible.append(i)
    return feasible


def action_to_X(a_idx, R, W_t, K=10):
    """Convert discrete action index to full decision dict.

    The discrete action only specifies (x_gb, x_bd).
    Solar allocation and grid-to-demand are computed greedily:
    solar goes to demand first (free!), surplus to battery.
    """
    x_gb, x_bd = ACTIONS[a_idx]
    x_bd = min(x_bd, R)
    x_gb = min(x_gb, K - R + x_bd)
    s = W_t['solar']
    d = W_t['demand']
    remaining_demand = max(0, d - x_bd)
    x_sd = min(s, remaining_demand)
    x_sb = min(s - x_sd, K - R - x_gb + x_bd)
    x_sb = max(0, x_sb)
    x_gd = max(0, remaining_demand - x_sd)
    return {'x_gb': x_gb, 'x_sb': x_sb, 'x_sd': x_sd,
            'x_bd': x_bd, 'x_gd': x_gd}


# Representative W values for each (price_regime, demand_regime)
REGIME_W = {
    (0, 0): {'price': 0.08, 'demand': 2.0, 'solar': 2.0},
    (0, 1): {'price': 0.08, 'demand': 3.0, 'solar': 2.0},
    (0, 2): {'price': 0.08, 'demand': 4.0, 'solar': 2.0},
    (1, 0): {'price': 0.12, 'demand': 2.0, 'solar': 2.0},
    (1, 1): {'price': 0.12, 'demand': 3.0, 'solar': 2.0},
    (1, 2): {'price': 0.12, 'demand': 4.0, 'solar': 2.0},
    (2, 0): {'price': 0.16, 'demand': 2.0, 'solar': 2.0},
    (2, 1): {'price': 0.16, 'demand': 3.0, 'solar': 2.0},
    (2, 2): {'price': 0.16, 'demand': 4.0, 'solar': 2.0},
}


def phi(R, pr, K=10):
    """Feature vector for linear VFA: [1, R/K, p/0.16, R·p/(K·0.16)]."""
    p_map = {0: 0.08, 1: 0.12, 2: 0.16}
    p = p_map.get(pr, 0.12)
    return np.array([1.0, R / K, p / 0.16, (R / K) * (p / 0.16)])


# ═══════════════════════════════════════════════════════════════
# Transition Probability Estimation
# ═══════════════════════════════════════════════════════════════

def estimate_transition_matrices(generate_W_fn, world_params,
                                 n_samples=100):
    """Estimate regime transition matrices from simulated data.

    Returns (price_trans, demand_trans), each 3×3.
    """
    price_trans = np.zeros((3, 3))
    demand_trans = np.zeros((3, 3))

    for seed in range(n_samples):
        W = generate_W_fn(world_params, seed=seed)
        for t in range(world_params.T - 1):
            pr_t = discretize_price(W['price'][t])
            pr_t1 = discretize_price(W['price'][t + 1])
            price_trans[pr_t, pr_t1] += 1
            dr_t = discretize_demand(W['demand'][t])
            dr_t1 = discretize_demand(W['demand'][t + 1])
            demand_trans[dr_t, dr_t1] += 1

    price_trans /= price_trans.sum(axis=1, keepdims=True) + 1e-10
    demand_trans /= demand_trans.sum(axis=1, keepdims=True) + 1e-10
    return price_trans, demand_trans


# ═══════════════════════════════════════════════════════════════
# Value Iteration
# ═══════════════════════════════════════════════════════════════

def value_iteration(price_trans, demand_trans,
                    K=10, gamma=0.95, tol=1e-6, max_iter=500):
    """Value iteration on the discretized Voltown MDP.

    Returns (V_star, pi_star, convergence_history).
    """
    STATES_local, S_idx_local = build_state_space(K)
    n_states = len(STATES_local)
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    history = []

    for it in range(max_iter):
        V_new = np.copy(V)
        for s_idx, (R, pr, dr) in enumerate(STATES_local):
            W_t = REGIME_W[(pr, dr)]
            best_val = np.inf
            best_a = 0

            for a in feasible_actions(R, K):
                X = action_to_X(a, R, W_t, K)
                c = period_cost(X, W_t)
                R_next = int(np.clip(transition(R, X, K), 0, K))

                future = 0.0
                for pr2 in range(3):
                    for dr2 in range(3):
                        s2_idx = S_idx_local[(R_next, pr2, dr2)]
                        future += (price_trans[pr, pr2]
                                   * demand_trans[dr, dr2]
                                   * V[s2_idx])

                q_val = c + gamma * future
                if q_val < best_val:
                    best_val = q_val
                    best_a = a

            V_new[s_idx] = best_val
            policy[s_idx] = best_a

        delta = np.max(np.abs(V_new - V))
        history.append(delta)
        V = V_new
        if delta < tol:
            break

    return V, policy, history


# ═══════════════════════════════════════════════════════════════
# Q-Learning
# ═══════════════════════════════════════════════════════════════

def q_learning(generate_W_fn, world_params,
               n_episodes=500, gamma=0.95,
               alpha0=0.1, epsilon0=0.3, seed=0):
    """Tabular Q-learning on discretized Voltown.

    Model-free, off-policy: learns Q* by ε-greedy exploration.

    Returns (Q_table, cost_history, visit_counts).
    """
    K = world_params.K
    STATES_local, S_idx_local = build_state_space(K)
    n_states = len(STATES_local)
    T = world_params.T

    rng = np.random.RandomState(seed)
    Q = np.zeros((n_states, N_ACTIONS))

    # Initialize infeasible state-actions to large cost
    for s_idx, (R, pr, dr) in enumerate(STATES_local):
        feas = feasible_actions(R, K)
        for a in range(N_ACTIONS):
            if a not in feas:
                Q[s_idx, a] = 1e6

    cost_history = []
    visits = np.zeros_like(Q)

    for ep in range(n_episodes):
        W_ep = generate_W_fn(world_params, seed=seed + ep)
        R = world_params.R_init
        total_cost = 0
        epsilon = epsilon0 / (1 + ep * 0.01)
        alpha = alpha0 / (1 + ep * 0.003)

        for t in range(T):
            W_t = {k: v[t] for k, v in W_ep.items()}
            R_disc = int(np.clip(round(R), 0, K))
            pr = discretize_price(W_t['price'])
            dr = discretize_demand(W_t['demand'])
            s_idx = S_idx_local[(R_disc, pr, dr)]
            feas = feasible_actions(R_disc, K)

            # ε-greedy action selection
            if rng.random() < epsilon:
                a = rng.choice(feas)
            else:
                q_feas = [(Q[s_idx, a], a) for a in feas]
                a = min(q_feas, key=lambda x: x[0])[1]

            X = action_to_X(a, R_disc, W_t, K)
            c = period_cost(X, W_t)
            R_next = transition(R_disc, X, K)
            total_cost += c

            R_next_disc = int(np.clip(round(R_next), 0, K))
            if t + 1 < T:
                W_next = {k: v[t + 1] for k, v in W_ep.items()}
                pr2 = discretize_price(W_next['price'])
                dr2 = discretize_demand(W_next['demand'])
                s2_idx = S_idx_local[(R_next_disc, pr2, dr2)]
                feas2 = feasible_actions(R_next_disc, K)
                min_q_next = min(Q[s2_idx, a2] for a2 in feas2)
            else:
                min_q_next = 0

            # Q-learning update: off-policy (uses min over next actions)
            td_target = c + gamma * min_q_next
            Q[s_idx, a] += alpha * (td_target - Q[s_idx, a])
            visits[s_idx, a] += 1
            R = R_next

        cost_history.append(total_cost)

    return Q, cost_history, visits


# ═══════════════════════════════════════════════════════════════
# ADP (Approximate Dynamic Programming)
# ═══════════════════════════════════════════════════════════════

def adp_train(generate_W_fn, world_params,
              n_episodes=300, gamma=0.95, alpha0=0.01, seed=0):
    """Train linear VFA via semi-gradient TD (ADP).

    Learns weights w for V_hat(s; w) = w^T · phi(s).

    Returns (weights, cost_history).
    """
    K = world_params.K
    T = world_params.T
    rng = np.random.RandomState(seed)
    w = np.zeros(4)
    cost_history = []

    for ep in range(n_episodes):
        W_ep = generate_W_fn(world_params, seed=seed + ep)
        R = world_params.R_init
        total_cost = 0
        alpha = alpha0 / (1 + ep * 0.005)

        for t in range(T):
            W_t = {k: v[t] for k, v in W_ep.items()}
            pr = discretize_price(W_t['price'])
            R_disc = int(np.clip(round(R), 0, K))

            # Approximate greedy action
            best_a, best_q = 0, np.inf
            for a in feasible_actions(R_disc, K):
                X = action_to_X(a, R_disc, W_t, K)
                c = period_cost(X, W_t)
                R_next = int(np.clip(transition(R_disc, X, K), 0, K))
                pr_next = pr  # assume regime persists (simplification)
                q = c + gamma * phi(R_next, pr_next, K) @ w
                if q < best_q:
                    best_q, best_a = q, a

            X = action_to_X(best_a, R_disc, W_t, K)
            c = period_cost(X, W_t)
            R_next = transition(R_disc, X, K)
            R_next_disc = int(np.clip(round(R_next), 0, K))
            total_cost += c

            # TD update
            pr_next = discretize_price(
                W_ep['price'][t + 1] if t + 1 < T else W_t['price'])
            v_curr = phi(R_disc, pr, K) @ w
            v_next = phi(R_next_disc, pr_next, K) @ w if t + 1 < T else 0
            td_error = c + gamma * v_next - v_curr
            w -= alpha * td_error * (-phi(R_disc, pr, K))

            R = R_next

        cost_history.append(total_cost)

    return w, cost_history
