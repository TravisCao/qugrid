"""Export QuGrid problems to Qiskit.

Qubit conventions match: QuGrid variable ``i`` is Qiskit qubit ``i``
(little-endian). Measured bitstrings from Qiskit print qubit 0 rightmost;
reverse them before calling ``problem.decode``, or use the helper here.
"""

from __future__ import annotations

import numpy as np

from qugrid.problems.base import CombinatorialProblem


def _require_qiskit():
    try:
        import qiskit  # noqa: F401
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "qiskit is not installed; install the extra with "
            "`pip install qugrid[qiskit]`"
        ) from err


def to_qiskit_operator(problem: CombinatorialProblem):
    """Ising cost Hamiltonian as a ``SparsePauliOp`` (identity term included)."""
    _require_qiskit()
    from qiskit.quantum_info import SparsePauliOp

    ising = problem.qubo.to_ising()
    n = ising.n
    terms: list[tuple[str, list[int], float]] = []
    if ising.offset:
        terms.append(("I", [0], float(ising.offset)))
    for i in range(n):
        if ising.h[i]:
            terms.append(("Z", [i], float(ising.h[i])))
    for i in range(n):
        for j in range(i + 1, n):
            if ising.j[i, j]:
                terms.append(("ZZ", [i, j], float(ising.j[i, j])))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def to_qiskit_qaoa(problem: CombinatorialProblem, p: int = 2):
    """Parameterized QAOA circuit for the problem.

    Returns ``(circuit, gammas, betas)`` where the parameter vectors bind the
    ``p`` cost and mixer angles. Ready for Qiskit primitives, transpilers,
    and hardware backends.
    """
    _require_qiskit()
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    ising = problem.qubo.to_ising()
    n = ising.n
    gammas = ParameterVector("gamma", p)
    betas = ParameterVector("beta", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if ising.j[i, j]]
    for layer in range(p):
        for i in range(n):
            if ising.h[i]:
                qc.rz(2.0 * float(ising.h[i]) * gammas[layer], i)
        for i, j in pairs:
            qc.rzz(2.0 * float(ising.j[i, j]) * gammas[layer], i, j)
        qc.rx(2.0 * betas[layer], range(n))
    qc.measure_all()
    return qc, gammas, betas


def bits_from_qiskit_key(key: str) -> np.ndarray:
    """Convert a Qiskit counts key (qubit 0 rightmost) to a QuGrid bit array."""
    return np.array([int(ch) for ch in reversed(key.replace(" ", ""))], dtype=int)
