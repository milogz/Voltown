# Voltown Sequential Decision Engine
# -----------------------------------
# A compact MDP environment for energy storage under uncertainty,
# structured around Warren Powell's unified framework.
#
# Modules:
#   world.py        — Exogenous information generator (W)
#   mdp.py          — Transition and cost functions
#   forecast.py     — Naive forecast engine
#   policies.py     — All policy implementations (PFA, CFA, VFA, DLA)
#   vfa_training.py — Value iteration, Q-learning, ADP training
#   simulator.py    — Episode runner and tournament engine
