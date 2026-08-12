"""A small algebra for assembling QUBOs from named variables and penalties.

Writing ``Q`` matrices by hand is where QUBO papers hide their bugs. The
builder keeps formulations readable and auditable::

    b = QUBOBuilder()
    u = [b.var(f"u_{g}") for g in range(3)]
    for g, cap in enumerate(caps):
        b.add_linear(u[g], cost[g])
    b.add_squared_penalty([(u[g], caps[g]) for g in range(3)], -demand, weight=lam)
    qubo = b.build()

``add_squared_penalty(terms, constant, weight)`` adds
``weight * (sum_i coef_i * x_i + constant)^2`` expanded exactly.
"""

from __future__ import annotations

import numpy as np

from qugrid.problems.base import QUBO


class QUBOBuilder:
    def __init__(self) -> None:
        self._names: list[str] = []
        self._index: dict[str, int] = {}
        self._lin: dict[int, float] = {}
        self._quad: dict[tuple[int, int], float] = {}
        self._const: float = 0.0

    def var(self, name: str) -> int:
        """Register a binary variable and return its index (idempotent)."""
        if name not in self._index:
            self._index[name] = len(self._names)
            self._names.append(name)
        return self._index[name]

    @property
    def n(self) -> int:
        return len(self._names)

    def add_constant(self, c: float) -> None:
        self._const += float(c)

    def add_linear(self, i: int, c: float) -> None:
        self._lin[i] = self._lin.get(i, 0.0) + float(c)

    def add_quadratic(self, i: int, j: int, c: float) -> None:
        if i == j:
            # x_i^2 == x_i for binaries
            self.add_linear(i, c)
            return
        key = (min(i, j), max(i, j))
        self._quad[key] = self._quad.get(key, 0.0) + float(c)

    def add_squared_penalty(
        self, terms: list[tuple[int, float]], constant: float, weight: float
    ) -> None:
        """Add ``weight * (sum coef_i x_i + constant)^2``, expanded exactly."""
        w = float(weight)
        c = float(constant)
        self.add_constant(w * c * c)
        for i, a in terms:
            self.add_linear(i, w * (a * a + 2.0 * a * c))
        for k in range(len(terms)):
            i, a = terms[k]
            for m in range(k + 1, len(terms)):
                j, b = terms[m]
                self.add_quadratic(i, j, 2.0 * w * a * b)

    def build(self) -> QUBO:
        q = np.zeros((self.n, self.n))
        for i, c in self._lin.items():
            q[i, i] += c
        for (i, j), c in self._quad.items():
            q[i, j] += c / 2.0
            q[j, i] += c / 2.0
        return QUBO(q=q, offset=self._const, names=list(self._names))
