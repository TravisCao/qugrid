"""Classical dispatch references: economic dispatch and enumerated unit commitment.

These provide the *true* optima against which QuGrid measures quantum
solutions — including the error introduced by QUBO discretization itself,
which is reported separately from solver error.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class GenParams:
    """One thermal generator for dispatch/commitment studies.

    Cost is quadratic: ``c2 * P^2 + c1 * P + c0`` [$ per hour], with ``c0``
    charged only when the unit is committed. ``startup`` is charged on each
    off-to-on transition.
    """

    name: str
    pmin: float
    pmax: float
    c2: float
    c1: float
    c0: float = 0.0
    startup: float = 0.0

    def cost(self, p: float | np.ndarray) -> float | np.ndarray:
        return self.c2 * p**2 + self.c1 * p + self.c0


def economic_dispatch(
    gens: list[GenParams], demand: float, tol: float = 1e-9
) -> tuple[np.ndarray, float]:
    """Exact single-period economic dispatch via bisection on marginal cost.

    Solves ``min sum_i c_i(P_i)  s.t.  sum_i P_i = demand, pmin_i <= P_i <= pmax_i``
    for strictly convex costs (``c2 > 0``). Returns ``(P, total_cost)``.
    Raises ``ValueError`` when the demand is outside the feasible range.
    """
    pmin = np.array([g.pmin for g in gens])
    pmax = np.array([g.pmax for g in gens])
    c1 = np.array([g.c1 for g in gens])
    c2 = np.array([g.c2 for g in gens])
    if np.any(c2 <= 0):
        raise ValueError("economic_dispatch requires strictly convex costs (c2 > 0)")
    if not (pmin.sum() - tol <= demand <= pmax.sum() + tol):
        raise ValueError(
            f"demand {demand:.6g} MW outside feasible range "
            f"[{pmin.sum():.6g}, {pmax.sum():.6g}] MW"
        )

    def output(lam: float) -> np.ndarray:
        return np.clip((lam - c1) / (2 * c2), pmin, pmax)

    lo = float(np.min(c1 + 2 * c2 * pmin))
    hi = float(np.max(c1 + 2 * c2 * pmax))
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        total = output(lam).sum()
        if abs(total - demand) < tol:
            break
        if total < demand:
            lo = lam
        else:
            hi = lam
    p = output(0.5 * (lo + hi))
    # Absorb the residual rounding into an interior (unclamped) unit.
    residual = demand - p.sum()
    interior = np.flatnonzero((p > pmin + tol) & (p < pmax - tol))
    if interior.size and abs(residual) > 0:
        p[interior[0]] += residual
    cost = float(sum(g.cost(p[i]) for i, g in enumerate(gens)))
    return p, cost


@dataclass
class UCSolution:
    """A unit commitment schedule with its dispatch and cost."""

    commit: np.ndarray  # (n_gen, n_periods) of 0/1
    power: np.ndarray  # (n_gen, n_periods) [MW]
    cost: float
    feasible: bool


def solve_uc_enumerate(
    gens: list[GenParams],
    demand: list[float] | np.ndarray,
    initial_on: np.ndarray | None = None,
) -> UCSolution:
    """Exact unit commitment by enumerating all commitment patterns.

    For each of the ``2^(G*T)`` on/off patterns, the committed units are
    dispatched with exact economic dispatch. Intended for the small instances
    (``G*T <= ~20``) used to validate quantum solvers; it is the ground truth,
    not a production UC.
    """
    demand = np.asarray(demand, dtype=float)
    n_gen, n_t = len(gens), len(demand)
    if n_gen * n_t > 24:
        raise ValueError("enumeration limited to G*T <= 24 variables")
    prev0 = np.zeros(n_gen) if initial_on is None else np.asarray(initial_on, dtype=float)

    best_cost = np.inf
    best: UCSolution | None = None
    for bits in product((0, 1), repeat=n_gen * n_t):
        commit = np.array(bits, dtype=float).reshape(n_gen, n_t)
        cost = 0.0
        power = np.zeros((n_gen, n_t))
        feasible = True
        for t in range(n_t):
            on = np.flatnonzero(commit[:, t])
            committed = [gens[i] for i in on]
            if not committed:
                feasible = False
                break
            try:
                p, c = economic_dispatch(committed, float(demand[t]))
            except ValueError:
                feasible = False
                break
            power[on, t] = p
            cost += c
            prev = prev0 if t == 0 else commit[:, t - 1]
            starts = np.maximum(commit[:, t] - prev, 0.0)
            cost += float(sum(g.startup * s for g, s in zip(gens, starts)))
        if feasible and cost < best_cost:
            best_cost = cost
            best = UCSolution(commit=commit, power=power, cost=cost, feasible=True)
    if best is None:
        raise ValueError("no feasible commitment pattern for the given demand")
    return best
