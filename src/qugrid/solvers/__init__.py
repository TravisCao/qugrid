"""Solver registry and the :func:`qugrid.solve` front door."""

from __future__ import annotations

from typing import Any

from qugrid.adapters.external_solvers import (
    solve_dimod_exact,
    solve_dwave_sa,
    solve_qiskit_qaoa,
)
from qugrid.problems.base import CombinatorialProblem, LinearSystemProblem
from qugrid.solvers.annealing import solve_sa
from qugrid.solvers.base import Result
from qugrid.solvers.exact import solve_exact_qubo, solve_linear_exact, solve_random
from qugrid.solvers.hhl import solve_hhl
from qugrid.solvers.kernel import (
    KernelRidgeClassifier,
    compare_kernels,
    quantum_kernel,
    rbf_kernel,
    scale_features,
)
from qugrid.solvers.qaoa import solve_qaoa
from qugrid.solvers.qbm import QuantumBoltzmannMachine
from qugrid.solvers.vqe import solve_vqe
from qugrid.solvers.vqls import solve_vqls

#: name -> (solver function, problem kind)
REGISTRY: dict[str, tuple[Any, str]] = {
    "exact": (solve_exact_qubo, "qubo"),
    "random": (solve_random, "qubo"),
    "sa": (solve_sa, "qubo"),
    "qaoa": (solve_qaoa, "qubo"),
    "vqe": (solve_vqe, "qubo"),
    "numpy": (solve_linear_exact, "linear"),
    "hhl": (solve_hhl, "linear"),
    "vqls": (solve_vqls, "linear"),
    # external stacks (optional extras); same Result, same metrics
    "dimod-exact": (solve_dimod_exact, "qubo"),
    "dwave-sa": (solve_dwave_sa, "qubo"),
    "qiskit-qaoa": (solve_qiskit_qaoa, "qubo"),
}


def solve(problem, solver: str = "auto", **kwargs) -> Result:
    """Solve any QuGrid problem with any registered solver.

    >>> import qugrid as qg
    >>> prob = qg.problems.Islanding(qg.cases.case9())
    >>> res = qg.solve(prob, solver="qaoa", seed=0)
    >>> print(res.summary())

    ``solver="auto"`` picks a sensible default: QAOA for small combinatorial
    problems, simulated annealing beyond statevector reach; HHL for small
    linear systems, exact algebra beyond.

    QUBO problems also accept the external stacks — ``"dimod-exact"``,
    ``"dwave-sa"``, ``"qiskit-qaoa"`` — which return the same
    :class:`Result` and need the matching extra installed.
    """
    if isinstance(problem, CombinatorialProblem):
        kind = "qubo"
        if solver == "auto":
            solver = "qaoa" if problem.n <= 16 else "sa"
    elif isinstance(problem, LinearSystemProblem):
        kind = "linear"
        if solver == "auto":
            solver = "hhl" if problem.n <= 16 else "numpy"
    else:
        raise TypeError(
            f"solve() expects a CombinatorialProblem or LinearSystemProblem, "
            f"got {type(problem).__name__}"
        )
    if solver not in REGISTRY:
        options = ", ".join(name for name, (_, k) in REGISTRY.items() if k == kind)
        raise ValueError(f"unknown solver {solver!r}; options for this problem: {options}")
    fn, expected = REGISTRY[solver]
    if expected != kind:
        options = ", ".join(name for name, (_, k) in REGISTRY.items() if k == kind)
        raise ValueError(
            f"solver {solver!r} does not apply to {type(problem).__name__}; "
            f"options: {options}"
        )
    return fn(problem, **kwargs)


__all__ = [
    "solve",
    "REGISTRY",
    "Result",
    "solve_exact_qubo",
    "solve_linear_exact",
    "solve_random",
    "solve_sa",
    "solve_qaoa",
    "solve_vqe",
    "solve_hhl",
    "solve_vqls",
    "solve_dimod_exact",
    "solve_dwave_sa",
    "solve_qiskit_qaoa",
    "quantum_kernel",
    "rbf_kernel",
    "scale_features",
    "compare_kernels",
    "KernelRidgeClassifier",
    "QuantumBoltzmannMachine",
]
