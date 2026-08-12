"""Export QuGrid problems to PennyLane."""

from __future__ import annotations

from qugrid.problems.base import CombinatorialProblem


def to_pennylane_hamiltonian(problem: CombinatorialProblem):
    """Ising cost Hamiltonian as a PennyLane operator (autodiff-ready).

    Use it in any PennyLane QNode, for example with
    ``qml.qaoa.cost_layer`` / ``qml.qaoa.mixer_layer``.
    """
    try:
        import pennylane as qml
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "pennylane is not installed; install the extra with "
            "`pip install qugrid[pennylane]`"
        ) from err

    ising = problem.qubo.to_ising()
    coeffs: list[float] = []
    ops: list = []
    if ising.offset:
        coeffs.append(float(ising.offset))
        ops.append(qml.Identity(0))
    for i in range(ising.n):
        if ising.h[i]:
            coeffs.append(float(ising.h[i]))
            ops.append(qml.PauliZ(i))
    for i in range(ising.n):
        for j in range(i + 1, ising.n):
            if ising.j[i, j]:
                coeffs.append(float(ising.j[i, j]))
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, ops)
