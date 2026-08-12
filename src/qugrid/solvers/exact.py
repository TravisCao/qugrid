"""Exact solvers: enumeration for QUBOs, dense algebra for linear systems.

The ground truth everything else is judged against.
"""

from __future__ import annotations

import numpy as np

from qugrid.problems.base import CombinatorialProblem, LinearSystemProblem
from qugrid.solvers.base import Result, attach_reference, finish_combinatorial, timed


def solve_exact_qubo(problem: CombinatorialProblem, **_ignored) -> Result:
    """Enumerate all ``2^n`` states (n <= 24) and return the optimum."""
    res = Result(solver="exact", problem=problem)
    with timed(res.resources):
        energies = problem.qubo.to_ising().all_energies()
        k = int(np.argmin(energies))
        x = problem.qubo.bits_from_index(k)
    finish_combinatorial(res, problem, x)
    res.resources["states_enumerated"] = len(energies)
    res.reference = {"objective": float(energies[k])}
    return res


def solve_linear_exact(problem: LinearSystemProblem, **_ignored) -> Result:
    """``numpy.linalg.solve`` — the classical reference for linear problems."""
    res = Result(solver="numpy", problem=problem)
    with timed(res.resources):
        x = problem.solve_exact()
    res.x = x
    residual = float(np.linalg.norm(problem.a @ x - problem.b))
    res.objective = residual
    res.decoded = {"residual_norm": residual}
    res.feasible = True
    res.resources["dimension"] = problem.n
    res.reference = {"x": x, "objective": 0.0}
    return res


def solve_random(problem: CombinatorialProblem, seed: int = 0, samples: int = 1000, **_) -> Result:
    """Uniform random sampling — the baseline any quantum claim must beat."""
    rng = np.random.default_rng(seed)
    res = Result(solver="random", problem=problem)
    with timed(res.resources):
        n = problem.qubo.n
        xs = rng.integers(0, 2, size=(samples, n))
        energies = problem.qubo.energy(xs)
        best = int(np.argmin(energies))
    finish_combinatorial(res, problem, xs[best])
    res.resources["samples"] = samples
    return attach_reference(res, problem)
