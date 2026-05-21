"""
Exogenous Information Generator — The "World" (W)
==================================================
Generates ground-truth sequences of demand, price, and solar
availability for the Voltown energy storage problem.

Powell's element:  W_{t+1} = (D_{t+1}, p_{t+1}, s_{t+1})
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class WorldParams:
    """Parameters that define the exogenous environment.

    These control the *ground truth* stochastic process — the "Nature"
    that generates realizations.  Students calibrate their models (W_hat)
    from samples of this process.
    """
    T: int = 24              # Periods per episode (e.g., hours in a day)
    K: float = 10.0          # Battery capacity (kWh)
    R_init: float = 2.0      # Initial battery level (kWh)
    cycles: int = 2          # Sinusoidal peaks per episode

    # Demand parameters
    demand_base: float = 3.0
    demand_amp: float = 1.5
    demand_noise: float = 0.4

    # Grid price parameters
    price_base: float = 0.12
    price_amp: float = 0.04
    price_noise: float = 0.008

    # Solar parameters
    solar_amp: float = 4.0
    solar_noise: float = 0.4


def generate_W(params: WorldParams, seed: int = 42) -> dict:
    """Generate one episode of exogenous information.

    Returns a dict with keys 'demand', 'price', 'solar',
    each a numpy array of length params.T.

    The sinusoidal base creates predictable daily patterns;
    Gaussian noise adds uncertainty.  This is intentionally simple —
    the pedagogical focus is on policy design, not forecasting.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(params.T)

    # Demand (kWh): sinusoidal + noise, clipped to stay positive
    demand = (params.demand_base
              + params.demand_amp * np.sin(params.cycles * 2 * np.pi * t / params.T))
    demand += rng.normal(0, params.demand_noise, params.T)
    demand = np.maximum(demand, 0.5)

    # Grid price ($/kWh): peaks when demand peaks
    price = (params.price_base
             + params.price_amp * np.sin(params.cycles * 2 * np.pi * t / params.T))
    price += rng.normal(0, params.price_noise, params.T)
    price = np.maximum(price, 0.02)

    # Solar availability (kWh): bell curve during "daylight" hours
    solar_base = params.solar_amp * np.sin(np.pi * (t - 4) / 16)
    solar = np.maximum(0, solar_base + rng.normal(0, params.solar_noise, params.T))

    return {'demand': demand, 'price': price, 'solar': solar}


def generate_W_extremistan(params: WorldParams, seed: int = 42) -> dict:
    """Generate an episode with heavy-tailed shocks (Extremistan).

    Same base process as generate_W, but with:
    - 10% chance of demand spike (2x normal)
    - 5% chance of price shock (3x normal)
    - 8% chance of solar blackout (near zero)

    This models the "tail risk" scenarios from Modules 1 and 2.
    """
    W = generate_W(params, seed)
    rng = np.random.RandomState(seed + 9999)

    # Demand spikes
    spike_mask = rng.random(params.T) < 0.10
    W['demand'][spike_mask] *= (1.5 + rng.random(spike_mask.sum()))

    # Price shocks
    price_shock = rng.random(params.T) < 0.05
    W['price'][price_shock] *= (2.0 + rng.random(price_shock.sum()) * 2)

    # Solar blackouts
    blackout = rng.random(params.T) < 0.08
    W['solar'][blackout] *= 0.1

    return W
