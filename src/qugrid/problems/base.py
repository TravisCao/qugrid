"""Problem abstractions: QUBO, Ising, and linear systems.

QuGrid separates three layers:

1. **Power problem** (unit commitment, islanding, ...): domain objects with
   engineering units and constraints.
2. **Mathematical encoding** (this module): quadratic unconstrained binary
   optimization (QUBO), its Ising form, or a linear system.
3. **Solver** (classical or quantum): consumes only the encoding.

Every quantum computer or annealer that exists today consumes layer 2, which
is why the encodings here are exact, tested, and convertible to Qiskit,
D-Wave, and PennyLane objects via :mod:`qugrid.adapters`.

Conventions
-----------
* QUBO: minimize ``x^T Q x + offset`` over ``x in {0,1}^n`` (``Q`` symmetric).
* Ising: minimize ``sum_{i<j} J_ij s_i s_j + sum_i h_i s_i + offset`` over
  ``s in {-1,+1}^n``, related by ``s = 1 - 2x`` (bit 0 maps to spin +1).
* Bit order: variable ``i`` is bit ``i`` of the integer basis-state index
  (little-endian), matching the statevector layout in
  :mod:`qugrid.solvers.statevector`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Ising:
    """Ising Hamiltonian ``H(s) = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i + offset``."""

    h: np.ndarray  # (n,)
    j: np.ndarray  # (n, n), strictly upper triangular (J_ij defined for i < j)
    offset: float = 0.0

    def __post_init__(self) -> None:
        # Normalize once so every consumer (SA local fields, adapters, QAOA)
        # can rely on the strictly-upper-triangular invariant.
        self.j = np.triu(np.asarray(self.j, dtype=float), k=1)
        self.h = np.asarray(self.h, dtype=float)

    @property
    def n(self) -> int:
        return len(self.h)

    def energy(self, s: np.ndarray) -> float | np.ndarray:
        """Energy of spin configuration(s); ``s`` is (n,) or (batch, n) of +-1."""
        s = np.asarray(s, dtype=float)
        if s.ndim == 1:
            return float(s @ self.j @ s + self.h @ s + self.offset)
        return np.einsum("bi,ij,bj->b", s, self.j, s) + s @ self.h + self.offset

    def all_energies(self) -> np.ndarray:
        """Energies of all ``2^n`` configurations (index bit ``i`` = variable ``i``).

        Memory stays at O(2^n) — spins are generated per variable, not cached
        per variable, so n = 24 costs ~400 MB peak instead of ~3 GB.
        """
        if self.n > 24:
            raise ValueError(f"refusing to enumerate 2^{self.n} states")
        states = np.arange(2**self.n, dtype=np.int64)

        def spin(i: int) -> np.ndarray:
            return 1.0 - 2.0 * ((states >> i) & 1)  # bit 0 -> spin +1

        energies = np.full(2**self.n, self.offset, dtype=float)
        for i in range(self.n):
            if self.h[i] != 0.0:
                energies += self.h[i] * spin(i)
        ii, jj = np.nonzero(self.j)
        last_a, s_a = -1, None
        for a, b in zip(ii, jj):
            if a != last_a:  # ii is row-sorted; cache the left spin per row
                s_a, last_a = spin(a), a
            energies += (self.j[a, b] * s_a) * spin(b)
        return energies


@dataclass
class QUBO:
    """Quadratic unconstrained binary optimization: minimize ``x^T Q x + offset``."""

    q: np.ndarray  # (n, n); symmetrized on construction
    offset: float = 0.0
    names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float)
        self.q = 0.5 * (q + q.T)
        if not self.names:
            self.names = [f"x{i}" for i in range(self.n)]

    @property
    def n(self) -> int:
        return self.q.shape[0]

    def energy(self, x: np.ndarray) -> float | np.ndarray:
        """Objective value of bitstring(s); ``x`` is (n,) or (batch, n) of 0/1."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return float(x @ self.q @ x + self.offset)
        return np.einsum("bi,ij,bj->b", x, self.q, x) + self.offset

    def to_ising(self) -> Ising:
        """Exact change of variables ``s = 1 - 2x``. Energies are preserved."""
        d = np.diag(self.q).copy()
        off = self.q - np.diag(d)  # symmetric, zero diagonal
        row = off.sum(axis=1)
        j = np.triu(off / 2.0, k=1)  # pair (i<j) contributes 2*Q_ij*x_i*x_j -> J_ij = Q_ij/2
        h = -row / 2.0 - d / 2.0
        offset = self.offset + off.sum() / 4.0 + d.sum() / 2.0
        return Ising(h=h, j=j, offset=offset)

    def bits_from_index(self, k: int) -> np.ndarray:
        return np.array([(k >> i) & 1 for i in range(self.n)], dtype=int)

    def dynamic_range(self, db: bool = False) -> float:
        """``max|coef| / min nonzero |coef|`` over the energy coefficients.

        Coefficients are the linear terms ``Q_ii`` and the pair couplings
        ``2 Q_ij`` (i < j); the offset shifts every energy equally and is
        excluded. Penalty-folded formulations stretch this ratio, and
        sampling solvers stop resolving the costs underneath: the 2-unit
        unit commitment study's penalized QUBO measures 3.9e3 here, and
        example 07 documents the QAOA success-probability collapse on it.
        ``db=True`` returns ``20 log10`` of the ratio. A QUBO with no
        nonzero coefficient reports 1 (0 dB).
        """
        lin = np.abs(np.diag(self.q))
        pairs = 2.0 * np.abs(self.q[np.triu_indices(self.n, k=1)])
        coefs = np.concatenate([lin, pairs])
        nonzero = coefs[coefs > 0.0]
        if nonzero.size == 0:
            return 0.0 if db else 1.0
        ratio = float(nonzero.max() / nonzero.min())
        return float(20.0 * np.log10(ratio)) if db else ratio


class CombinatorialProblem(ABC):
    """A power system decision problem encoded as a QUBO.

    Subclasses formulate the power problem, own its constraints, and know how
    to translate a raw bitstring back into an engineering answer.
    """

    @property
    @abstractmethod
    def qubo(self) -> QUBO:
        """The QUBO encoding (constraints folded in as quadratic penalties)."""

    @abstractmethod
    def decode(self, x: np.ndarray) -> dict:
        """Translate a bitstring into engineering quantities."""

    @abstractmethod
    def is_feasible(self, x: np.ndarray) -> bool:
        """Check the original (pre-penalty) constraints."""

    def reference(self) -> dict:
        """Classical reference optimum (exhaustive over the QUBO by default)."""
        energies = self.qubo.to_ising().all_energies()
        k = int(np.argmin(energies))
        x = self.qubo.bits_from_index(k)
        out = self.decode(x)
        out["objective"] = float(energies[k])
        out["x"] = x
        return out

    @property
    def n(self) -> int:
        return self.qubo.n

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={self.n} binary variables)"


@dataclass
class LinearSystemProblem:
    """A linear system ``A x = b`` destined for a (quantum) linear solver.

    ``scale`` and ``names`` carry the engineering meaning of ``x`` so that
    solvers can hand back answers in physical units.
    """

    a: np.ndarray
    b: np.ndarray
    names: list[str] = field(default_factory=list)
    unit: str = ""
    context: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        if not self.names:
            self.names = [f"x{i}" for i in range(len(self.b))]

    @property
    def n(self) -> int:
        return len(self.b)

    @property
    def is_hermitian(self) -> bool:
        return bool(np.allclose(self.a, self.a.T))

    def condition_number(self) -> float:
        return float(np.linalg.cond(self.a))

    def solve_exact(self) -> np.ndarray:
        return np.linalg.solve(self.a, self.b)

    def padded(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Embed into the next power-of-two dimension (identity padding).

        Quantum registers hold ``2^m`` amplitudes; the padding is inert
        (``A'`` block diagonal with the identity, ``b'`` zero on the pad).
        Returns ``(A', b', original_dimension)``.
        """
        n = self.n
        m = 1 if n <= 1 else int(np.ceil(np.log2(n)))
        full = 2**m
        a = np.eye(full)
        a[:n, :n] = self.a
        b = np.zeros(full)
        b[:n] = self.b
        return a, b, n

    def __repr__(self) -> str:
        return (
            f"LinearSystemProblem(n={self.n}, unit={self.unit!r}, "
            f"cond={self.condition_number():.3g})"
        )
