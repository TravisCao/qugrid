"""Optimal PMU placement as a QUBO (minimum dominating set).

A phasor measurement unit (PMU) at a bus observes that bus and every adjacent
bus. Full topological observability with the fewest PMUs is the minimum
dominating set of the network graph — NP-hard, small enough on benchmark
systems to verify exactly, and a standard entry point for quantum
optimization in power systems.

Coverage constraints ``x_i + sum_{j in N(i)} x_j >= 1`` become equalities via
binary slack variables before the penalty expansion, the textbook route for
inequality constraints on annealers (Lucas, "Ising formulations of many NP
problems", Front. Physics 2014, arXiv:1302.5843).
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from qugrid.network import Network
from qugrid.problems.base import QUBO, CombinatorialProblem
from qugrid.problems.builder import QUBOBuilder


class PMUPlacement(CombinatorialProblem):
    """Minimum-PMU full observability (zero-injection buses not modeled).

    Variables: one placement bit per bus plus ``ceil(log2(deg_i + 1))`` slack
    bits per bus. The IEEE 14-bus system needs 14 + 26 = 40 binaries — small
    for an annealer, instructive for counting why qubit budgets matter.
    """

    def __init__(self, net: Network, penalty: float | None = None):
        self.net = net
        a = net.adjacency()
        self.neighbors = [np.flatnonzero(a[i]).tolist() for i in range(net.n_bus)]
        self.degrees = np.array([len(nb) for nb in self.neighbors])
        self.penalty = penalty if penalty is not None else float(2 * net.n_bus)

    @cached_property
    def qubo(self) -> QUBO:
        n = self.net.n_bus
        bld = QUBOBuilder()
        x = [bld.var(f"pmu@bus{int(self.net.bus[i, 0])}") for i in range(n)]
        for i in range(n):
            bld.add_linear(x[i], 1.0)  # objective: number of PMUs
        for i in range(n):
            # x_i + sum_{j in N(i)} x_j - 1 - slack_i = 0, slack_i in [0, deg_i]
            n_slack_bits = max(1, int(np.ceil(np.log2(self.degrees[i] + 1))))
            terms = [(x[i], 1.0)] + [(x[j], 1.0) for j in self.neighbors[i]]
            terms += [
                (bld.var(f"s[{i},{k}]"), -float(2**k)) for k in range(n_slack_bits)
            ]
            bld.add_squared_penalty(terms, -1.0, weight=self.penalty)
        return bld.build()

    def coverage(self, x: np.ndarray) -> np.ndarray:
        """Boolean per bus: observed by the placement bits in ``x``."""
        x = np.asarray(x, dtype=int)
        n = self.net.n_bus
        placed = x[:n].astype(bool)
        covered = placed.copy()
        for i in range(n):
            if placed[i]:
                covered[self.neighbors[i]] = True
        return covered

    def decode(self, x: np.ndarray) -> dict:
        n = self.net.n_bus
        placed = np.flatnonzero(np.asarray(x[:n], dtype=int))
        covered = self.coverage(x)
        return {
            "pmu_buses": [int(self.net.bus[i, 0]) for i in placed],
            "n_pmu": int(len(placed)),
            "n_uncovered": int((~covered).sum()),
            "covered": covered,
        }

    def is_feasible(self, x: np.ndarray) -> bool:
        return bool(self.coverage(x).all())

    def reference(self) -> dict:
        """Exact minimum dominating set by enumeration over placement bits."""
        n = self.net.n_bus
        if n > 20:
            raise ValueError("exact enumeration limited to 20 buses")
        best_x: np.ndarray | None = None
        best = n + 1
        for k in range(2**n):
            x = np.array([(k >> i) & 1 for i in range(n)], dtype=int)
            if x.sum() >= best:
                continue
            if self.coverage(x).all():
                best = int(x.sum())
                best_x = x
        assert best_x is not None
        out = self.decode(best_x)
        out["x"] = best_x
        return out
