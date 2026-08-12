"""Security screening datasets for quantum machine learning.

Operators screen thousands of operating points for limit violations; machine
learning classifiers are studied as fast surrogates for that screening. This
module builds small, physically grounded classification datasets from any
:class:`~qugrid.network.Network`, labeled by exact DC power flow — a clean
benchmark for quantum kernels versus classical baselines, with no synthetic
blobs pretending to be power systems.

The default label is **N-1 security**: an operating point is insecure when any
single-branch outage (among outages that keep the grid connected) pushes some
branch loading above ``threshold``. That is the screening task operators
actually run — IEEE test cases are built to be secure in the base topology, so
base-case labels would be all zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qugrid import idx
from qugrid.classical.power_flow import solve_dc
from qugrid.network import Network


@dataclass
class ScreeningDataset:
    """Features, labels, and provenance for a screening task."""

    x: np.ndarray  # (n_samples, n_features) load multipliers
    y: np.ndarray  # (n_samples,) 1 = insecure (some contingency loading > threshold)
    feature_names: list[str]
    threshold: float
    contingencies: list[int]  # branch indices screened (empty = base case only)
    net_name: str

    def split(self, train_fraction: float = 0.7, seed: int = 0):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.y))
        cut = int(train_fraction * len(self.y))
        tr, te = order[:cut], order[cut:]
        return self.x[tr], self.y[tr], self.x[te], self.y[te]


def _connected_without(net: Network, branch: int) -> bool:
    """Is the network still one island after removing ``branch``?"""
    adj: list[set[int]] = [set() for _ in range(net.n_bus)]
    for k, (f, t, _w) in enumerate(net.edges()):
        if k != branch:
            adj[f].add(t)
            adj[t].add(f)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == net.n_bus


def screenable_contingencies(net: Network) -> list[int]:
    """Branch indices whose outage keeps the network connected (non-bridges)."""
    return [k for k in range(net.n_branch) if _connected_without(net, k)]


def screening_dataset(
    net: Network,
    n_samples: int = 200,
    load_range: tuple[float, float] = (0.6, 1.4),
    threshold: float = 1.0,
    contingencies: list[int] | str | None = "n-1",
    seed: int = 0,
) -> ScreeningDataset:
    """Sample load patterns and label them secure/insecure by DC power flow.

    Each load bus gets an independent multiplier drawn uniformly from
    ``load_range``; generation is scaled proportionally to keep balance.
    The label is 1 when the maximum branch loading (|flow| / RATE_A) exceeds
    ``threshold`` in the base case **or in any screened contingency**.

    ``contingencies``: ``"n-1"`` (default) screens every single-branch outage
    that keeps the grid connected; ``None`` labels on the base case only; a
    list of branch indices screens exactly those outages.

    Features are the multipliers — low-dimensional (one per load bus) and
    physically meaningful, sized for today's quantum feature maps.
    """
    if not np.any(net.branch[:, idx.RATE_A] > 0):
        raise ValueError("network has no branch ratings (RATE_A); cannot label security")
    if contingencies == "n-1":
        cont = screenable_contingencies(net)
    elif contingencies is None:
        cont = []
    else:
        cont = list(contingencies)
        for k in cont:
            if not _connected_without(net, k):
                raise ValueError(f"outage of branch {k} disconnects the network")
    topologies = [net.copy()] + [net.drop_branch(k) for k in cont]

    load_buses = np.flatnonzero(net.bus[:, idx.PD] > 0)
    rng = np.random.default_rng(seed)
    x = rng.uniform(load_range[0], load_range[1], size=(n_samples, len(load_buses)))
    y = np.zeros(n_samples, dtype=int)
    base_load = net.bus[:, idx.PD].sum()
    for s in range(n_samples):
        worst = 0.0
        for topo in topologies:
            sample = topo.copy()
            factors = np.ones(sample.n_bus)
            factors[load_buses] = x[s]
            sample.bus[:, idx.PD] *= factors
            sample.bus[:, idx.QD] *= factors
            # keep generation-load balance by scaling PG proportionally
            scale = sample.bus[:, idx.PD].sum() / max(base_load, 1e-9)
            sample.gen[:, idx.PG] *= scale
            worst = max(worst, solve_dc(sample).max_loading())
            if worst > threshold:
                break
        y[s] = int(worst > threshold)
    names = [f"load_bus{int(net.bus[i, idx.BUS_I])}" for i in load_buses]
    return ScreeningDataset(
        x=x,
        y=y,
        feature_names=names,
        threshold=threshold,
        contingencies=cont,
        net_name=net.name,
    )
