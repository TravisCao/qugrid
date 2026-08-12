"""A minimal, exact statevector simulator in pure NumPy.

QuGrid ships its own simulator so that ``pip install qugrid`` runs quantum
algorithms with zero quantum SDK installed. It is deliberately small:

* States are flat complex arrays of length ``2^n``; qubit ``i`` is bit ``i``
  of the basis-state index (little-endian), the same convention as
  :meth:`qugrid.problems.base.Ising.all_energies`.
* Only the gates the built-in algorithms need. For anything else, export the
  problem to Qiskit or PennyLane via :mod:`qugrid.adapters`.
* Exact expectation values by default; finite-shot sampling on request.

Practical ceiling: ~20 qubits (a 16 MB state). Every entry point checks and
says so instead of freezing your laptop.
"""

from __future__ import annotations

import numpy as np

MAX_QUBITS = 22


def check_size(n: int, what: str = "statevector simulation") -> None:
    if n > MAX_QUBITS:
        raise ValueError(
            f"{what} needs {n} qubits; the built-in simulator caps at {MAX_QUBITS}. "
            "Use solver='sa' for a classical baseline at this size, or export via "
            "qugrid.adapters to run on external simulators/hardware."
        )


def zero_state(n: int) -> np.ndarray:
    check_size(n)
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    return psi


def uniform_state(n: int) -> np.ndarray:
    check_size(n)
    return np.full(2**n, 2 ** (-n / 2), dtype=complex)


def _as3(psi: np.ndarray, n: int, q: int) -> np.ndarray:
    return psi.reshape(2 ** (n - q - 1), 2, 2**q)


def apply_rx_all(psi: np.ndarray, n: int, beta: float) -> np.ndarray:
    """exp(-i * beta * X) on every qubit (the QAOA mixer with angle beta)."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(n):
        v = _as3(psi, n, q)
        a = v[:, 0, :].copy()
        b = v[:, 1, :].copy()
        v[:, 0, :] = c * a + s * b
        v[:, 1, :] = c * b + s * a
    return psi


def apply_ry(psi: np.ndarray, n: int, q: int, theta: float) -> np.ndarray:
    """Standard Ry(theta) rotation on qubit ``q``."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    v = _as3(psi, n, q)
    a = v[:, 0, :].copy()
    b = v[:, 1, :].copy()
    v[:, 0, :] = c * a - s * b
    v[:, 1, :] = s * a + c * b
    return psi


def apply_h_all(psi: np.ndarray, n: int) -> np.ndarray:
    inv = 1 / np.sqrt(2)
    for q in range(n):
        v = _as3(psi, n, q)
        a = v[:, 0, :].copy()
        b = v[:, 1, :].copy()
        v[:, 0, :] = inv * (a + b)
        v[:, 1, :] = inv * (a - b)
    return psi


def apply_cz_chain(psi: np.ndarray, n: int) -> np.ndarray:
    """CZ between neighbors (0,1), (1,2), ... — the entangler of the built-in ansatz."""
    idx = np.arange(len(psi))
    for q in range(n - 1):
        mask = ((idx >> q) & (idx >> (q + 1)) & 1).astype(bool)
        psi[mask] *= -1.0
    return psi


def diag_phase(psi: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Multiply by ``exp(1j * phases)`` elementwise (any diagonal unitary)."""
    psi *= np.exp(1j * phases)
    return psi


def probabilities(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi) ** 2


def expectation_diag(psi: np.ndarray, diag: np.ndarray) -> float:
    """Exact expectation value of a diagonal observable."""
    return float(np.real(np.dot(probabilities(psi), diag)))


def sample_counts(psi: np.ndarray, shots: int, rng: np.random.Generator) -> dict[int, int]:
    p = probabilities(psi)
    p = p / p.sum()
    draws = rng.choice(len(p), size=shots, p=p)
    states, counts = np.unique(draws, return_counts=True)
    return {int(s): int(c) for s, c in zip(states, counts)}


def bits_of(state: int, n: int) -> np.ndarray:
    """Little-endian bits of a basis-state index (bit i = variable/qubit i)."""
    return np.array([(state >> i) & 1 for i in range(n)], dtype=int)


def real_amplitudes_ansatz(params: np.ndarray, n: int) -> np.ndarray:
    """Hardware-efficient real-amplitude ansatz: Ry layers with CZ-chain entanglers.

    ``params`` has shape ``(layers + 1, n)``. This mirrors Qiskit's
    ``RealAmplitudes`` circuit, so trained parameters transfer.
    """
    params = np.asarray(params, dtype=float).reshape(-1, n)
    psi = zero_state(n)
    for layer, row in enumerate(params):
        if layer > 0:
            apply_cz_chain(psi, n)
        for q in range(n):
            apply_ry(psi, n, q, row[q])
    return psi
