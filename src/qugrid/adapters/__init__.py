"""Bridges into the quantum ecosystems researchers already use.

QuGrid's built-in simulator is for zero-setup research; when you want shots,
noise models, annealing hardware, or autodiff, export the *same* problem:

* :func:`to_qiskit_operator` / :func:`to_qiskit_qaoa` — IBM Qiskit
* :func:`to_bqm` / :func:`result_from_sampleset` — D-Wave Ocean (dimod)
* :func:`to_pennylane_hamiltonian` — Xanadu PennyLane

The same bridges also run in the other direction: :mod:`external solvers
<qugrid.adapters.external_solvers>` hand the problem to a vendor solver and
return the standard :class:`~qugrid.solvers.base.Result`, so
``qg.solve(problem, solver="dwave-sa")`` reads like any built-in solver.

Each function imports its ecosystem lazily and raises a clear message naming
the extra to install (``pip install qugrid[qiskit]`` and friends).
"""

from qugrid.adapters.dimod_adapter import result_from_sampleset, to_bqm
from qugrid.adapters.external_solvers import (
    solve_dimod_exact,
    solve_dwave_sa,
    solve_qiskit_qaoa,
)
from qugrid.adapters.pennylane_adapter import to_pennylane_hamiltonian
from qugrid.adapters.qiskit_adapter import to_qiskit_operator, to_qiskit_qaoa

__all__ = [
    "to_qiskit_operator",
    "to_qiskit_qaoa",
    "to_bqm",
    "result_from_sampleset",
    "to_pennylane_hamiltonian",
    "solve_dimod_exact",
    "solve_dwave_sa",
    "solve_qiskit_qaoa",
]
