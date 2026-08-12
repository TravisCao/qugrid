"""Economic dispatch as a QUBO via binary power expansion.

Single-period dispatch of committed units. The continuous optimum is cheap
classically (see :func:`qugrid.classical.dispatch.economic_dispatch`); the
QUBO version exists to quantify what discretization costs and to give
variational solvers a small, well-understood target with a known answer.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from qugrid.classical.dispatch import GenParams, economic_dispatch
from qugrid.problems.base import QUBO, CombinatorialProblem
from qugrid.problems.builder import QUBOBuilder


class EconomicDispatchQUBO(CombinatorialProblem):
    """Minimize generation cost subject to a demand-balance penalty.

    Output of unit ``g``: ``P_g = pmin_g + delta_g * sum_k 2^k b_{g,k}`` with
    ``delta_g = (pmax - pmin) / (2^K - 1)``. All units are committed.
    """

    def __init__(
        self,
        gens: list[GenParams],
        demand: float,
        power_bits: int = 3,
        weight_balance: float | None = None,
    ) -> None:
        self.gens = list(gens)
        self.demand = float(demand)
        self.power_bits = int(power_bits)
        self.delta = np.array(
            [(g.pmax - g.pmin) / (2**self.power_bits - 1) for g in self.gens]
        )
        cost_ceiling = float(sum(g.cost(g.pmax) for g in self.gens))
        min_quantum = float(self.delta.min())
        self.weight_balance = (
            weight_balance
            if weight_balance is not None
            else 2.0 * cost_ceiling / max(min_quantum**2, 1e-9)
        )
        self.balance_tol = 0.75 * min_quantum

    @cached_property
    def qubo(self) -> QUBO:
        bld = QUBOBuilder()
        balance_terms: list[tuple[int, float]] = []
        offset = 0.0
        for g, gen in enumerate(self.gens):
            bits = [bld.var(f"b[{g},{k}]") for k in range(self.power_bits)]
            p_terms = [(bits[k], self.delta[g] * 2**k) for k in range(self.power_bits)]
            # cost of (pmin + sum) : expand c2*(pmin + s)^2 + c1*(pmin + s) + c0
            bld.add_squared_penalty(p_terms, gen.pmin, weight=gen.c2)
            for i, coef in p_terms:
                bld.add_linear(i, gen.c1 * coef)
            offset += gen.c1 * gen.pmin + gen.c0  # constant cost terms (all units on)
            balance_terms.extend(p_terms)
        total_pmin = float(sum(g.pmin for g in self.gens))
        bld.add_squared_penalty(
            balance_terms, total_pmin - self.demand, weight=self.weight_balance
        )
        bld.add_constant(offset)
        return bld.build()

    def power(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=int)
        p = np.zeros(len(self.gens))
        for g, gen in enumerate(self.gens):
            level = sum(
                2**k * x[g * self.power_bits + k] for k in range(self.power_bits)
            )
            p[g] = gen.pmin + self.delta[g] * level
        return p

    def decode(self, x: np.ndarray) -> dict:
        p = self.power(x)
        cost = float(sum(g.cost(p[i]) for i, g in enumerate(self.gens)))
        return {
            "power": p,
            "cost": cost,
            "balance_error_mw": float(abs(p.sum() - self.demand)),
        }

    def is_feasible(self, x: np.ndarray) -> bool:
        return bool(abs(self.power(x).sum() - self.demand) <= self.balance_tol)

    def continuous_reference(self) -> dict:
        p, cost = economic_dispatch(self.gens, self.demand)
        return {"power": p, "cost": cost}
