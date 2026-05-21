"""Quick integration test for the Voltown engine."""
import sys
sys.path.insert(0, '.')

from engine.world import WorldParams, generate_W, generate_W_extremistan
from engine.mdp import transition, period_cost
from engine.forecast import forecast_naive
import numpy as np

# 1. World generator
wp = WorldParams()
W = generate_W(wp, 42)
print(f"[OK] World: T={len(W['demand'])}, demand_mean={np.mean(W['demand']):.2f}")

We = generate_W_extremistan(wp, 42)
print(f"[OK] Extremistan: max_price={np.max(We['price']):.4f}")

# 2. MDP core
X = {'x_gb': 2.0, 'x_sb': 1.0, 'x_sd': 1.0, 'x_bd': 0.5, 'x_gd': 1.0}
W_t = {'price': 0.12, 'demand': 3.0, 'solar': 2.0}
print(f"[OK] Transition: R_next={transition(2.0, X, 10.0):.1f}")
print(f"[OK] Cost: {period_cost(X, W_t):.4f}")

# 3. Forecast
fc = forecast_naive(
    {'demand': W['demand'][:5], 'price': W['price'][:5], 'solar': W['solar'][:5]},
    H=4, num_paths=3)
print(f"[OK] Forecast: shape={fc['demand'].shape}")

# 4. VFA training
from engine.vfa_training import (build_state_space, feasible_actions,
                                  action_to_X, estimate_transition_matrices,
                                  value_iteration, q_learning)

STATES, S_idx = build_state_space(10)
print(f"[OK] State space: |S|={len(STATES)}")
print(f"[OK] Feasible(R=2): {feasible_actions(2, 10)}")

price_trans, demand_trans = estimate_transition_matrices(generate_W, wp, n_samples=20)
print(f"[OK] Transition matrices estimated")

V, pi, hist = value_iteration(price_trans, demand_trans, K=10, max_iter=100)
print(f"[OK] Value Iteration: converged in {len(hist)} iters, V range=[{V.min():.3f}, {V.max():.3f}]")

# 5. Policies (PFA, CFA)
from engine.policies import policy_PFA_threshold, policy_CFA

W0 = {k: v[0] for k, v in W.items()}
X_pfa = policy_PFA_threshold(2.0, W0, {'K': 10.0, 'theta1': 0.12, 'theta2': 8.0, 'theta3': 5.0})
print(f"[OK] PFA: x_gb={X_pfa['x_gb']:.2f}, x_gd={X_pfa['x_gd']:.2f}")

X_cfa = policy_CFA(2.0, W0, {'K': 10.0, 'theta': 0.10})
print(f"[OK] CFA: x_gb={X_cfa['x_gb']:.2f}, x_gd={X_cfa['x_gd']:.2f}")

# 6. Simulator
from engine.simulator import run_episode
r = run_episode(policy_PFA_threshold, W, 
                {'K': 10.0, 'theta1': 0.12, 'theta2': 8.0, 'theta3': 5.0},
                K=10.0, R_init=2.0)
print(f"[OK] Episode: total_cost=${r['total_cost']:.4f}, T={len(r['costs'])}")

print("\n✅ ALL ENGINE TESTS PASSED")
