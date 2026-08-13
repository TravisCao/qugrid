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
        self._n_ineq: int = 0

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

    def add_inequality(
        self,
        terms: list[tuple[int, float]],
        constant: float,
        weight: float = 1.0,
        method: str = "slack",
        lam: tuple[float, float] | None = None,
        name: str | None = None,
    ) -> None:
        """Penalize the constraint ``sum_i coef_i * x_i + constant >= 0``.

        ``method="slack"`` (default) is the textbook route (Lucas,
        arXiv:1302.5843): register binary slack bits covering ``[0, U]``
        with ``U`` the maximum of the left side over all bitstrings, and add
        ``weight * (sum_i coef_i x_i + constant - S)^2``. Exact whenever the
        left side is integer-valued; costs ``ceil(log2(U + 1))`` extra bits
        per constraint.

        ``method="unbalanced"`` is unbalanced penalization
        (Montanez-Barrera et al., Quantum Sci. Technol. 9, 025022, 2024;
        arXiv:2211.13914): add ``-l1*g + l2*g^2`` for
        ``g = sum_i coef_i x_i + constant``, with ``lam=(l1, l2)``. Zero
        slack bits, so the qubit count stops growing with the number of
        inequalities — but the encoding is a heuristic: the unconstrained
        minimum coincides with the constrained optimum only for suitable
        ``(l1, l2)``, which need tuning per problem family. ``weight`` is
        ignored on this path.
        """
        if name is None:
            name = f"ineq{self._n_ineq}"
        self._n_ineq += 1
        if method == "slack":
            g_max = float(constant) + sum(max(a, 0.0) for _, a in terms)
            u = int(round(g_max))
            if abs(g_max - u) > 1e-9:
                raise ValueError("slack encoding needs an integer-valued left side")
            if u < 0:
                raise ValueError("constraint can never hold: left side is at most negative")
            n_bits = max(1, int(np.ceil(np.log2(u + 1)))) if u > 0 else 0
            slack = [(self.var(f"{name}_s{k}"), -float(2**k)) for k in range(n_bits)]
            self.add_squared_penalty(list(terms) + slack, constant, weight)
        elif method == "unbalanced":
            l1, l2 = lam if lam is not None else (1.0, 1.0)
            self.add_constant(-l1 * float(constant))
            for i, a in terms:
                self.add_linear(i, -l1 * a)
            self.add_squared_penalty(list(terms), constant, weight=l2)
        else:
            raise ValueError(f"unknown method {method!r}; use 'slack' or 'unbalanced'")

    def build(self) -> QUBO:
        q = np.zeros((self.n, self.n))
        for i, c in self._lin.items():
            q[i, i] += c
        for (i, j), c in self._quad.items():
            q[i, j] += c / 2.0
            q[j, i] += c / 2.0
        return QUBO(q=q, offset=self._const, names=list(self._names))
