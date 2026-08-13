"""The :class:`Result` object every QuGrid solver returns.

One result type across classical baselines, simulators, and hardware
adapters, built for the two questions researchers actually ask:

* *How good is the answer?* — engineering quantities, feasibility, and the
  gap to the classical reference, never just a raw bitstring.
* *What did it cost?* — qubits, parameters, evaluations, wall time.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Result:
    """Outcome of solving a QuGrid problem."""

    solver: str
    problem: Any
    x: np.ndarray | None = None
    objective: float | None = None
    decoded: dict = field(default_factory=dict)
    feasible: bool | None = None
    history: list[float] = field(default_factory=list)
    top_states: list[tuple[str, float, float]] = field(default_factory=list)
    # (bitstring, probability, energy) for combinatorial quantum solvers
    resources: dict = field(default_factory=dict)
    reference: dict | None = None
    extras: dict = field(default_factory=dict)

    # ------------------------------------------------------------ derived
    def gap(self) -> float | None:
        """Relative optimality gap versus the reference objective (if known)."""
        if self.reference is None or self.objective is None:
            return None
        ref = self.reference.get("objective")
        if ref is None:
            return None
        denom = max(abs(ref), 1e-12)
        return float((self.objective - ref) / denom)

    def success_probability(self) -> float | None:
        """Probability mass on the reference optimum (quantum solvers only)."""
        if self.reference is None or not self.top_states:
            return None
        ref = self.reference.get("objective")
        if ref is None:
            return None
        tol = 1e-9 * max(1.0, abs(ref))
        return float(
            sum(p for _, p, e in self.top_states if abs(e - ref) <= tol)
        )

    # ------------------------------------------------------------- display
    def summary(self) -> str:
        lines = [f"QuGrid result | solver={self.solver} | {self._problem_name()}"]
        if self.objective is not None:
            lines.append(f"  objective          {self.objective:,.6g}")
        if self.feasible is not None:
            lines.append(f"  feasible           {'yes' if self.feasible else 'NO'}")
        g = self.gap()
        if g is not None:
            if abs(g) < 1e-9:  # display float noise as the exact match it is
                g = 0.0
            lines.append(f"  gap vs reference   {100 * g:.3g}%")
        sp = self.success_probability()
        if sp is not None:
            lines.append(f"  P(optimum)         {sp:.3f}")
        pr = self.extras.get("p_optimum_repaired")
        if pr is not None:
            lines.append(f"  P(opt | repaired)  {pr:.3f}")
        for key, val in self.decoded.items():
            if isinstance(val, (int, float, np.floating)):
                lines.append(f"  {key:<18} {val:,.6g}")
            elif isinstance(val, tuple) and all(isinstance(v, (int, float)) for v in val):
                shown = tuple(v if isinstance(v, (bool, int)) else round(float(v), 4) for v in val)
                lines.append(f"  {key:<18} {shown}")
        if self.resources:
            res = ", ".join(f"{k}={v}" for k, v in self.resources.items())
            lines.append(f"  resources          {res}")
        return "\n".join(lines)

    def _problem_name(self) -> str:
        p = self.problem
        name = type(p).__name__
        n = getattr(p, "n", None)
        return f"{name}(n={n})" if n is not None else name

    def plot_convergence(self, ax=None):
        """Optimizer trajectory (thin wrapper over :func:`qugrid.viz.plot_convergence`)."""
        from qugrid.viz import plot_convergence

        return plot_convergence(self, ax=ax)

    def __repr__(self) -> str:
        obj = f"{self.objective:.6g}" if self.objective is not None else "n/a"
        return f"Result(solver={self.solver!r}, objective={obj}, feasible={self.feasible})"


@contextmanager
def timed(resources: dict):
    """Record wall time into a resources dict."""
    t0 = time.perf_counter()
    yield
    resources["wall_time_s"] = round(time.perf_counter() - t0, 4)


def attach_reference(result: Result, problem) -> Result:
    """Attach the classical reference when the problem can produce one.

    Formulations may override ``reference()`` with problem-specific
    enumeration that stays cheap beyond the generic limit (PMU placement
    enumerates placement bits only, not slack bits). The default enumerates
    the full QUBO up to n = 24 and raises beyond; a missing reference is a
    convenience lost, never a failure.
    """
    try:
        result.reference = problem.reference()
    except Exception:
        result.reference = None
    return result


def finish_combinatorial(result: Result, problem, x: np.ndarray) -> Result:
    """Fill decoded/objective/feasible fields from a bitstring."""
    result.x = np.asarray(x, dtype=int)
    result.objective = float(problem.qubo.energy(result.x))
    result.decoded = problem.decode(result.x)
    result.feasible = problem.is_feasible(result.x)
    return result
