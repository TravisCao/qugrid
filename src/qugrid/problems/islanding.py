"""Controlled islanding as a QUBO.

Splitting a stressed system into self-sufficient islands is a classic
combinatorial problem in power system operation (and one of the first grid
problems solved on a quantum annealer). The QUBO trades off three terms:

* **Cut weight** — total electrical coupling ``1/x`` of the lines that are
  opened. Cutting strong ties is expensive.
* **Power imbalance** — squared generation-load mismatch inside each island;
  balanced islands can survive on their own.
* **Size balance** — a small term that rules out the degenerate "no split"
  answer and steers toward comparable island sizes.

Reference: H. Ushijima-Mwesigwa, C. F. A. Negre, S. M. Mniszewski,
"Graph Partitioning using Quantum Annealing on the D-Wave System", 2017
(arXiv:1705.03082), adapted with a power-balance term.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from qugrid.network import Network
from qugrid.problems.base import QUBO, CombinatorialProblem
from qugrid.problems.builder import QUBOBuilder


class Islanding(CombinatorialProblem):
    """Two-way controlled islanding of a network.

    Variables: ``x_i = 0/1`` assigns bus ``i`` to island A or B.

    Parameters
    ----------
    net:
        The network. One binary variable per bus.
    alpha:
        Weight of the squared power-imbalance term (per-unit power). ``None``
        scales it so a one-per-unit imbalance costs as much as cutting the
        strongest line.
    beta:
        Weight of the size-balance term. ``None`` uses a small default that
        breaks the all-one-island degeneracy without dominating.
    """

    def __init__(self, net: Network, alpha: float | None = None, beta: float | None = None):
        self.net = net
        self.edge_list = net.edges()
        w = np.array([w for _, _, w in self.edge_list])
        self.injection_pu = (net.gen_p_per_bus() - net.load_p) / net.baseMVA
        self.alpha = alpha if alpha is not None else float(2.0 * w.max())
        self.beta = beta if beta is not None else float(0.05 * w.mean())

    @cached_property
    def qubo(self) -> QUBO:
        n = self.net.n_bus
        bld = QUBOBuilder()
        x = [bld.var(f"bus{int(self.net.bus[i, 0])}") for i in range(n)]
        # cut term: w * (x_f + x_t - 2 x_f x_t)
        for f, t, w in self.edge_list:
            bld.add_linear(x[f], w)
            bld.add_linear(x[t], w)
            bld.add_quadratic(x[f], x[t], -2.0 * w)
        # power imbalance: alpha * (sum_i p_i * s_i)^2 with s = 2x - 1
        terms = [(x[i], 2.0 * self.injection_pu[i]) for i in range(n)]
        const = -float(self.injection_pu.sum())
        bld.add_squared_penalty(terms, const, weight=self.alpha)
        # size balance: beta * (sum_i s_i)^2
        bld.add_squared_penalty([(x[i], 2.0) for i in range(n)], -float(n), weight=self.beta)
        return bld.build()

    # ---------------------------------------------------------------- decode
    def decode(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=int)
        cut = [(f, t) for f, t, _ in self.edge_list if x[f] != x[t]]
        cut_weight = float(sum(w for f, t, w in self.edge_list if x[f] != x[t]))
        p_a = float(self.injection_pu[x == 0].sum() * self.net.baseMVA)
        p_b = float(self.injection_pu[x == 1].sum() * self.net.baseMVA)
        return {
            "islands": x,
            "sizes": (int((x == 0).sum()), int((x == 1).sum())),
            "cut_lines": cut,
            "n_cut": len(cut),
            "cut_weight": cut_weight,
            "island_power_mw": (p_a, p_b),
            "islands_connected": self._connected(x),
        }

    def _connected(self, x: np.ndarray) -> tuple[bool, bool]:
        out = []
        for side in (0, 1):
            nodes = set(np.flatnonzero(x == side).tolist())
            if not nodes:
                out.append(False)
                continue
            adj: dict[int, list[int]] = {v: [] for v in nodes}
            for f, t, _ in self.edge_list:
                if f in nodes and t in nodes:
                    adj[f].append(t)
                    adj[t].append(f)
            start = next(iter(nodes))
            seen = {start}
            stack = [start]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            out.append(seen == nodes)
        return (out[0], out[1])

    def is_feasible(self, x: np.ndarray) -> bool:
        x = np.asarray(x, dtype=int)
        return bool(0 < x.sum() < len(x))
