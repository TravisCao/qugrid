"""Methods from the optimization literature, wrapped around QuGrid solvers.

The first entry is the augmented Lagrangian outer loop for
equality-constrained QUBOs. Fixed quadratic penalties force a choice between
two failure modes: too small and the solver returns infeasible bitstrings,
too large and the penalty stretches the QUBO dynamic range until sampling
solvers stop resolving the costs underneath (see
:meth:`qugrid.problems.QUBO.dynamic_range`). The augmented Lagrangian keeps
the quadratic weight small and moves constraint pressure into linear
multiplier terms instead, re-solving with any QuGrid solver between updates.

References: Hong, Xu, and Teng, "Quantum annealing-aided aggregated
unit commitment", arXiv:2502.15917 (stochastic UC on D-Wave, IEEE 118-bus);
Feng, Zhang, Bragin, and Zhou, "Scalability and performance of quantum
computing for unit commitment", IEEE Trans. Power Systems 38(3), 2023
(surrogate Lagrangian relaxation).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from qugrid.problems.base import QUBO, CombinatorialProblem
from qugrid.solvers.base import Result, attach_reference, finish_combinatorial, timed

#: One linear constraint ``sum_i coef_i x_i + constant = 0``:
#: ``([(index, coef), ...], constant)`` — the shape ``QUBOBuilder`` uses.
Constraint = tuple[list[tuple[int, float]], float]


def _dense(constraints: list[Constraint], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Constraint list as ``(A, c)`` with residuals ``A x + c``."""
    a = np.zeros((len(constraints), n))
    c = np.zeros(len(constraints))
    for row, (terms, const) in enumerate(constraints):
        for i, coef in terms:
            a[row, i] += coef
        c[row] = const
    return a, c


class ConstrainedQUBOProblem(CombinatorialProblem):
    """A cost QUBO plus explicit linear equality constraints.

    ``qubo`` is the *pure* objective — no penalty folded in. Feasibility is
    ``|A x + c| <= tol`` for every constraint row. The reference optimum
    enumerates the objective over the feasible set (n <= 20).
    """

    def __init__(self, objective: QUBO, constraints: list[Constraint], tol: float = 1e-6):
        self._qubo = objective
        self.constraints = constraints
        self.tol = float(tol)
        self._a, self._c = _dense(constraints, objective.n)

    @property
    def qubo(self) -> QUBO:
        return self._qubo

    def residuals(self, x: np.ndarray) -> np.ndarray:
        return self._a @ np.asarray(x, dtype=float) + self._c

    def decode(self, x: np.ndarray) -> dict:
        r = self.residuals(x)
        return {
            "x": np.asarray(x, dtype=int),
            "residuals": r,
            "max_violation": float(np.abs(r).max()) if r.size else 0.0,
        }

    def is_feasible(self, x: np.ndarray) -> bool:
        r = self.residuals(x)
        return bool(r.size == 0 or np.abs(r).max() <= self.tol)

    def reference(self) -> dict:
        if self.n > 20:
            raise ValueError("feasible enumeration limited to 20 variables")
        states = np.arange(2**self.n, dtype=np.int64)
        bits = ((states[:, None] >> np.arange(self.n)) & 1).astype(float)
        feasible = np.all(np.abs(bits @ self._a.T + self._c) <= self.tol, axis=1)
        if not feasible.any():
            raise ValueError("no feasible bitstring exists")
        energies = self.qubo.energy(bits[feasible])
        k = int(np.argmin(energies))
        x = bits[feasible][k].astype(int)
        out = self.decode(x)
        out["objective"] = float(energies[k])
        out["x"] = x
        return out


@dataclass
class ALIterate:
    """One outer iteration: the multipliers used and the iterate they produced."""

    iteration: int
    x: np.ndarray
    cost: float  # pure objective, no penalty
    residuals: np.ndarray
    max_violation: float
    feasible: bool
    lam: float
    u: np.ndarray


class AugmentedLagrangianLoop:
    """Outer loop ``L = f + sum_c u_c g_c + lam sum_c g_c^2`` over a QUBO solver.

    After each inner solve at ``x*``: ``u_c += 2 lam g_c(x*)`` and
    ``lam *= alpha``. The quadratic weight starts small and grows only until
    the iterate is feasible, so the QUBO dynamic range stays flat compared
    with the fixed-penalty encoding. Multiplier updates are the standard
    method-of-multipliers step (Hestenes/Powell); the references above apply
    it to power system QUBOs.

    Parameters
    ----------
    objective:
        The pure cost QUBO (no constraint penalty folded in).
    constraints:
        Linear equalities ``sum coef_i x_i + constant = 0`` in
        ``QUBOBuilder`` term form.
    solver:
        Any :func:`qugrid.solve` name for the inner solves.
    lam0, alpha, iters, tol:
        Initial quadratic weight, its growth factor, the outer iteration
        cap, and the feasibility tolerance on each residual.

    :meth:`run` returns the standard :class:`Result` of the best feasible
    iterate measured by the *pure* objective (``feasible=False`` on the
    least-violating iterate if none is feasible); ``self.history`` keeps the
    per-iteration :class:`ALIterate` records.
    """

    def __init__(
        self,
        objective: QUBO,
        constraints: list[Constraint],
        solver: str = "sa",
        lam0: float = 1.0,
        alpha: float = 2.0,
        iters: int = 10,
        tol: float = 1e-6,
    ) -> None:
        self.problem = ConstrainedQUBOProblem(objective, constraints, tol=tol)
        self.solver = solver
        self.lam0 = float(lam0)
        self.alpha = float(alpha)
        self.iters = int(iters)
        self.history: list[ALIterate] = []

    def run(self, seed: int = 0, **solver_kwargs) -> Result:
        from qugrid.solvers import solve  # local import; solvers pull in adapters

        prob = self.problem
        objective, a, c = prob.qubo, prob._a, prob._c
        aa = a.T @ a
        ca = 2.0 * (c @ a)
        u = np.zeros(len(c))
        lam = self.lam0
        self.history = []
        best: ALIterate | None = None

        result = Result(solver=f"al({self.solver})", problem=prob)
        with timed(result.resources):
            for k in range(self.iters):
                q = objective.q + lam * aa
                lin = u @ a + lam * ca
                q = q + np.diag(lin)
                offset = objective.offset + float(u @ c) + lam * float(c @ c)
                sub = ConstrainedQUBOProblem(
                    QUBO(q=q, offset=offset, names=list(objective.names)),
                    prob.constraints,
                    tol=prob.tol,
                )
                with warnings.catch_warnings():
                    # Late iterations stretch the penalized QUBO on purpose;
                    # the dynamic-range warning would misfire here.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    inner = solve(sub, solver=self.solver, seed=seed + k, **solver_kwargs)
                x = np.asarray(inner.x, dtype=int)
                g = a @ x + c
                it = ALIterate(
                    iteration=k,
                    x=x,
                    cost=float(objective.energy(x)),
                    residuals=g,
                    max_violation=float(np.abs(g).max()) if g.size else 0.0,
                    feasible=prob.is_feasible(x),
                    lam=lam,
                    u=u.copy(),
                )
                self.history.append(it)
                if it.feasible and (best is None or not best.feasible or it.cost < best.cost):
                    best = it
                elif best is None or (not best.feasible and it.max_violation < best.max_violation):
                    best = it
                if it.feasible and k > 0 and np.array_equal(x, self.history[k - 1].x):
                    break  # converged: same feasible iterate twice
                u = u + 2.0 * lam * g
                lam = lam * self.alpha
        assert best is not None
        finish_combinatorial(result, prob, best.x)
        result.resources.update(
            {
                "iterations": len(self.history),
                "lam_final": self.history[-1].lam,
                "max_violation": best.max_violation,
                "inner_solver": self.solver,
            }
        )
        return attach_reference(result, prob)
