"""Bridges into the quantum ecosystems researchers already use.

QuGrid's built-in simulator is for zero-setup research; when you want shots,
noise models, annealing hardware, or autodiff, export the *same* problem:

* :func:`to_qiskit_operator` / :func:`to_qiskit_qaoa` — IBM Qiskit
* :func:`to_bqm` / :func:`result_from_sampleset` — D-Wave Ocean (dimod)
* :func:`to_pennylane_hamiltonian` — Xanadu PennyLane

Each function imports its ecosystem lazily and raises a clear message naming
the extra to install (``pip install qugrid[qiskit]`` and friends).
"""

from qugrid.adapters.dimod_adapter import result_from_sampleset, to_bqm
from qugrid.adapters.pennylane_adapter import to_pennylane_hamiltonian
from qugrid.adapters.qiskit_adapter import to_qiskit_operator, to_qiskit_qaoa

__all__ = [
    "to_qiskit_operator",
    "to_qiskit_qaoa",
    "to_bqm",
    "result_from_sampleset",
    "to_pennylane_hamiltonian",
]
