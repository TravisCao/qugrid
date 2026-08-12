"""Toy renewable scenario data for generative-model studies.

Scenario generation — sampling plausible wind/solar trajectories — is where
quantum Boltzmann machines and quantum GANs enter power system research. This
module provides a small, seeded wind dataset with realistic structure
(diurnal shape, ramps, temporal correlation) plus the binarization utilities
that map profiles onto spins.
"""

from __future__ import annotations

import numpy as np


def toy_wind_profiles(
    n_profiles: int = 200,
    horizon: int = 6,
    seed: int = 0,
    ramp_prob: float = 0.35,
) -> np.ndarray:
    """Sample wind capacity-factor profiles in ``[0, 1]`` of shape ``(n, horizon)``.

    The generator mixes two regimes — steady output around a random level, and
    ramp events (front passages) that swing output monotonically — then adds
    AR(1) noise. Not a physical model; a distribution with learnable joint
    structure (means, ramps, neighbor correlations) for generative-model demos.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n_profiles, horizon))
    for i in range(n_profiles):
        if rng.random() < ramp_prob:
            lo, hi = np.sort(rng.uniform(0.05, 0.95, size=2))
            ramp = np.linspace(lo, hi, horizon)
            if rng.random() < 0.5:
                ramp = ramp[::-1]
            base = ramp
        else:
            base = np.full(horizon, rng.uniform(0.1, 0.9))
        noise = np.zeros(horizon)
        for t in range(1, horizon):
            noise[t] = 0.7 * noise[t - 1] + rng.normal(0, 0.06)
        out[i] = np.clip(base + noise, 0.0, 1.0)
    return out


def binarize(profiles: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Threshold profiles into 0/1 patterns (default: global median)."""
    thr = float(np.median(profiles)) if threshold is None else float(threshold)
    return (profiles > thr).astype(int)


def empirical_statistics(bits: np.ndarray) -> dict:
    """Mean activation per step and neighbor correlations — the statistics a
    scenario generator must reproduce."""
    means = bits.mean(axis=0)
    corr = np.array(
        [np.corrcoef(bits[:, t], bits[:, t + 1])[0, 1] for t in range(bits.shape[1] - 1)]
    )
    return {"means": means, "neighbor_corr": corr}
