"""
Forecast Engine — Naive Predictions for DLA Policies
=====================================================
Generates forecast scenarios for Direct Lookahead policies.

In Powell's framework, the forecast is what DLA uses to
"look ahead" — it's an approximation of future W.
The quality of the forecast directly affects DLA performance.
"""

import numpy as np


def forecast_naive(W_history: dict, H: int,
                   num_paths: int = 1, noise_std: float = 0.3) -> dict:
    """Naive forecast: historical mean + Gaussian noise for H periods.

    This is intentionally unsophisticated — we want students to see
    that DLA performance depends on forecast quality, and that even
    a naive forecast enables anticipation.

    Returns dict with keys 'demand', 'price', 'solar',
    each of shape (num_paths, H).
    """
    rng = np.random.RandomState()  # uses current numpy seed
    fc = {}
    for var in ['demand', 'price', 'solar']:
        hist = W_history[var]
        mu = np.mean(hist)
        sigma = np.std(hist) if len(hist) > 1 else noise_std
        paths = mu + sigma * rng.randn(num_paths, H)
        paths = np.maximum(paths, 0.01)
        fc[var] = paths
    return fc
