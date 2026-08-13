"""Unit commitment as a QUBO, with the discretization made explicit.

The formulation commits units *and* dispatches them, using a binary expansion
of each unit's output. All constraints enter as quadratic penalties:

* ``u_{g,t}``: unit ``g`` on in period ``t``.
* ``b_{g,t,k}``: k-th power bit; output ``P_{g,t} = pmin_g u_{g,t} +
  delta_g * sum_k 2^k b_{g,t,k}`` with ``delta_g = (pmax-pmin)/(2^K - 1)``.
* Balance: ``weight_balance * (sum_g P_{g,t} - D_t)^2`` per period.
* Gating: ``weight_gate * b_{g,t,k} (1 - u_{g,t})`` keeps power bits at zero
  for uncommitted units.
* Startup cost: ``su_g * u_{g,t} (1 - u_{g,t-1})`` — already quadratic.

Two error sources are reported separately: the *discretization gap* (QUBO
optimum vs continuous-dispatch optimum) and the *solver gap* (found solution
vs QUBO optimum). Conflating them is a common mistake in the literature.

Reference formulation: A. Koretsky et al., "Adapting Quantum Approximation
Optimization Algorithm (QAOA) for Unit Commitment", IEEE QCE 2021
(arXiv:2110.12624), extended here with startup costs and per-unit binary
dispatch.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from qugrid.classical.dispatch import GenParams, solve_uc_enumerate
from qugrid.problems.base import QUBO, CombinatorialProblem
from qugrid.problems.builder import QUBOBuilder


class UnitCommitment(CombinatorialProblem):
    """Multi-period unit commitment with binary-encoded dispatch.

    Parameters
    ----------
    gens:
        Generator parameters (cost coefficients in $/h, limits in MW).
    demand:
        Demand per period [MW].
    power_bits:
        Bits per unit and period for the dispatch expansion. ``2`` gives four
        output levels between ``pmin`` and ``pmax``.
    weight_balance, weight_gate:
        Penalty weights. ``None`` picks values that dominate the worst-case
        cost range, which is sufficient for exactness on feasible instances.
    initial_on:
        Commitment state before the first period (for startup costs).
    """

    def __init__(
        self,
        gens: list[GenParams],
        demand: list[float] | np.ndarray,
        power_bits: int = 2,
        weight_balance: float | None = None,
        weight_gate: float | None = None,
        initial_on: np.ndarray | None = None,
    ) -> None:
        self.gens = list(gens)
        self.demand = np.asarray(demand, dtype=float)
        self.power_bits = int(power_bits)
        if self.power_bits < 1:
            raise ValueError("power_bits must be >= 1")
        self.n_gen = len(self.gens)
        self.n_t = len(self.demand)
        self.initial_on = (
            np.zeros(self.n_gen) if initial_on is None else np.asarray(initial_on, float)
        )
        self.delta = np.array(
            [(g.pmax - g.pmin) / (2**self.power_bits - 1) for g in self.gens]
        )
        cost_ceiling = float(
            sum(g.cost(g.pmax) + g.startup for g in self.gens) * self.n_t
        )
        min_quantum = float(self.delta.min())
        self.weight_balance = (
            weight_balance
            if weight_balance is not None
            else 2.0 * cost_ceiling / max(min_quantum**2, 1e-9)
        )
        self.weight_gate = weight_gate if weight_gate is not None else 2.0 * cost_ceiling
        self.balance_tol = 0.75 * min_quantum

    # ------------------------------------------------------------- variables
    def _u(self, g: int, t: int) -> str:
        return f"u[{g},{t}]"

    def _b(self, g: int, t: int, k: int) -> str:
        return f"b[{g},{t},{k}]"

    @cached_property
    def qubo(self) -> QUBO:
        bld = QUBOBuilder()
        u = {(g, t): bld.var(self._u(g, t)) for g in range(self.n_gen) for t in range(self.n_t)}
        b = {
            (g, t, k): bld.var(self._b(g, t, k))
            for g in range(self.n_gen)
            for t in range(self.n_t)
            for k in range(self.power_bits)
        }

        for t in range(self.n_t):
            balance_terms: list[tuple[int, float]] = []
            for g, gen in enumerate(self.gens):
                # P_{g,t} as linear terms over (u, bits)
                p_terms = [(u[g, t], gen.pmin)] + [
                    (b[g, t, k], self.delta[g] * 2**k) for k in range(self.power_bits)
                ]
                balance_terms.extend(p_terms)
                # cost: c2 * P^2 + c1 * P + c0 * u
                bld.add_squared_penalty(p_terms, 0.0, weight=gen.c2)
                for i, coef in p_terms:
                    bld.add_linear(i, gen.c1 * coef)
                bld.add_linear(u[g, t], gen.c0)
                # startup: su * u_t * (1 - u_{t-1})
                if gen.startup:
                    if t == 0:
                        bld.add_linear(u[g, t], gen.startup * (1.0 - self.initial_on[g]))
                    else:
                        bld.add_linear(u[g, t], gen.startup)
                        bld.add_quadratic(u[g, t], u[g, t - 1], -gen.startup)
                # gating: bits force u on
                for k in range(self.power_bits):
                    bld.add_linear(b[g, t, k], self.weight_gate)
                    bld.add_quadratic(b[g, t, k], u[g, t], -self.weight_gate)
            bld.add_squared_penalty(balance_terms, -float(self.demand[t]), self.weight_balance)
        return bld.build()

    # --------------------------------------------------------------- decoding
    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        names = self.qubo.names
        commit = np.zeros((self.n_gen, self.n_t))
        power = np.zeros((self.n_gen, self.n_t))
        val = {name: int(x[i]) for i, name in enumerate(names)}
        for g in range(self.n_gen):
            for t in range(self.n_t):
                commit[g, t] = val[self._u(g, t)]
                level = sum(2**k * val[self._b(g, t, k)] for k in range(self.power_bits))
                power[g, t] = self.gens[g].pmin * commit[g, t] + self.delta[g] * level
        return commit, power

    def decode(self, x: np.ndarray) -> dict:
        commit, power = self._split(x)
        cost = 0.0
        for t in range(self.n_t):
            for g, gen in enumerate(self.gens):
                if commit[g, t]:
                    cost += float(gen.cost(power[g, t]))
                prev = self.initial_on[g] if t == 0 else commit[g, t - 1]
                cost += gen.startup * max(commit[g, t] - prev, 0.0)
        balance = power.sum(axis=0) - self.demand
        return {
            "commit": commit,
            "power": power,
            "cost": cost,
            "balance_error_mw": float(np.abs(balance).max()),
        }

    def is_feasible(self, x: np.ndarray) -> bool:
        commit, power = self._split(x)
        gate_ok = bool(np.all(power[commit == 0] == 0))
        balance = power.sum(axis=0) - self.demand
        return gate_ok and bool(np.abs(balance).max() <= self.balance_tol)

    def constraint_residual(self, x: np.ndarray) -> float:
        """MW of violation: imbalance beyond tolerance plus gated-off power."""
        commit, power = self._split(x)
        balance = np.abs(power.sum(axis=0) - self.demand)
        gate = float(power[commit == 0].sum())
        return float(np.maximum(balance - self.balance_tol, 0.0).sum() + gate)

    # -------------------------------------------------------------- baselines
    def continuous_reference(self) -> dict:
        """True UC optimum with continuous dispatch (enumeration + exact ED)."""
        sol = solve_uc_enumerate(self.gens, self.demand, initial_on=self.initial_on)
        return {"commit": sol.commit, "power": sol.power, "cost": sol.cost}
