"""QAOA mixers: transverse-field, warm-start, and XY (one-hot) mixing.

The mixer decides which states QAOA can reach and which it never leaves.
QuGrid's variants come from the quantum optimization literature with power
system applications attached:

* :class:`XMixer` — the standard transverse field of Farhi, Goldstone,
  Gutmann (arXiv:1411.4028). Explores the full hypercube from the uniform
  superposition.
* :class:`WarmStartMixer` — warm-start QAOA of Egger, Marecek, Woerner
  (Quantum 5, 479, 2021): start from a product state biased toward a
  classical relaxation and mix around it, so depth-1 circuits already
  concentrate near a good solution. Salgado, Sequeira, Santos
  (arXiv:2412.11312) apply it to unit commitment and reach 5.1% of the
  reference at p=1.
* :class:`XYMixer` — the XY-model mixer of Wang, Hadfield, Jiang, Rieffel
  (Phys. Rev. A 101, 012320, 2020): exact ring-XY evolution inside each
  one-hot group keeps the state in the Hamming-weight-1 subspace, so
  one-hot constraints hold with certainty and need no penalty at all.
  Mohseni et al. (arXiv:2603.00260) run the biased variant on IBM hardware
  for single-period unit commitment.

Every mixer implements ``initial_state(n)`` and ``apply(psi, n, beta)``;
:func:`qugrid.solvers.qaoa.solve_qaoa` accepts any of them via ``mixer=``.
"""

from __future__ import annotations

import numpy as np

from qugrid.solvers.statevector import (
    apply_rx_all,
    apply_ry,
    apply_rz,
    apply_unitary,
    uniform_state,
    zero_state,
)


class XMixer:
    """Transverse-field mixer: ``exp(-i beta X)`` per qubit from ``|+>^n``."""

    name = "x"

    def initial_state(self, n: int) -> np.ndarray:
        return uniform_state(n)

    def apply(self, psi: np.ndarray, n: int, beta: float) -> np.ndarray:
        return apply_rx_all(psi, n, beta)


class WarmStartMixer:
    """Warm-start mixer around a relaxed solution ``c`` in ``[0, 1]^n``.

    The initial state is the product ``R_y(theta_i)|0>`` with
    ``theta_i = 2 arcsin(sqrt(c_i))``, so qubit ``i`` measures 1 with
    probability ``c_i``. The mixer conjugates a ``Z`` rotation into the same
    frame — ``R_y(theta_i) R_z(-2 beta) R_y(-theta_i)`` — which keeps the
    initial state a mixer ground state (Egger, Marecek, Woerner, Quantum 5,
    479, 2021). ``epsilon`` clips ``c`` into ``[epsilon, 1 - epsilon]``:
    at 0 or 1 the mixer becomes diagonal and the bit freezes, so the clip
    is what keeps the search alive away from the relaxation.
    """

    name = "warm-start"

    def __init__(self, c: np.ndarray, epsilon: float = 0.25):
        c = np.clip(np.asarray(c, dtype=float), epsilon, 1.0 - epsilon)
        if not (0.0 <= epsilon <= 0.5):
            raise ValueError("epsilon must lie in [0, 0.5]")
        self.c = c
        self.epsilon = float(epsilon)
        self.thetas = 2.0 * np.arcsin(np.sqrt(c))

    def initial_state(self, n: int) -> np.ndarray:
        if len(self.thetas) != n:
            raise ValueError(f"warm start has {len(self.thetas)} entries for {n} qubits")
        psi = zero_state(n)
        for q, theta in enumerate(self.thetas):
            apply_ry(psi, n, q, theta)
        return psi

    def apply(self, psi: np.ndarray, n: int, beta: float) -> np.ndarray:
        for q, theta in enumerate(self.thetas):
            apply_ry(psi, n, q, -theta)
            apply_rz(psi, n, q, -2.0 * beta)
            apply_ry(psi, n, q, theta)
        return psi


def _xy_ring_hamiltonian(k: int) -> np.ndarray:
    """Ring-XY Hamiltonian ``sum_edges (X_a X_b + Y_a Y_b) / 2`` on k qubits."""
    edges = [(j, (j + 1) % k) for j in range(k)] if k > 2 else [(0, 1)]
    h = np.zeros((2**k, 2**k))
    for a, b in edges:
        for idx in range(2**k):
            if (idx >> a) & 1 and not (idx >> b) & 1:
                swapped = idx ^ (1 << a) ^ (1 << b)
                h[idx, swapped] += 1.0
                h[swapped, idx] += 1.0
    return h


class XYMixer:
    """Ring-XY mixer over one-hot groups, W-state initialized.

    ``groups`` lists the qubit indices of each one-hot set (for a discretized
    dispatch: one group per unit, one qubit per output level). Within a
    group the exact evolution ``exp(-i beta H_XY)`` only exchanges
    excitations, so the Hamming weight per group is conserved: start in the
    W state (weight 1) and every reachable state satisfies the one-hot
    constraint — no penalty term, no slack, no infeasible samples. Qubits
    outside every group get the standard transverse-field treatment.

    The honest cost is qubit count: one-hot needs ``levels x units`` qubits
    where binary needs ``ceil(log2(levels)) x units``. The 22-qubit
    statevector ceiling holds about 7 units at 3 levels.
    """

    name = "xy"

    def __init__(self, groups: list[list[int]]):
        self.groups = [list(g) for g in groups]
        flat = [q for g in self.groups for q in g]
        if len(set(flat)) != len(flat):
            raise ValueError("one-hot groups overlap")
        if any(len(g) < 2 for g in self.groups):
            raise ValueError("a one-hot group needs at least 2 qubits")
        self._eig = []
        for g in self.groups:
            w, v = np.linalg.eigh(_xy_ring_hamiltonian(len(g)))
            self._eig.append((w, v))

    def initial_state(self, n: int) -> np.ndarray:
        in_group = {q for g in self.groups for q in g}
        rest = [q for q in range(n) if q not in in_group]
        psi = zero_state(n)
        psi[0] = 0.0
        rest_offsets = np.array(
            [
                sum(((m >> j) & 1) << rest[j] for j in range(len(rest)))
                for m in range(2 ** len(rest))
            ]
        )
        amp = np.prod([1.0 / np.sqrt(len(g)) for g in self.groups]) * 2 ** (-len(rest) / 2)
        bases = np.array([0])
        for g in self.groups:  # W state per group: one hot qubit each
            bases = (bases[:, None] + np.array([1 << q for q in g])[None, :]).ravel()
        psi[(bases[:, None] + rest_offsets[None, :]).ravel()] = amp
        return psi

    def apply(self, psi: np.ndarray, n: int, beta: float) -> np.ndarray:
        in_group = {q for g in self.groups for q in g}
        for g, (w, v) in zip(self.groups, self._eig):
            u = (v * np.exp(-1j * beta * w)) @ v.T
            apply_unitary(psi, n, g, u)
        for q in range(n):
            if q not in in_group:
                # exp(-i beta X) on the leftover qubits, matching XMixer
                c, s = np.cos(beta), -1j * np.sin(beta)
                u = np.array([[c, s], [s, c]])
                apply_unitary(psi, n, [q], u)
        return psi
