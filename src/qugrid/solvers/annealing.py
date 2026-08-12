"""Simulated annealing — the honest classical baseline for QUBOs.

Any claim that a quantum optimizer helps a power system problem must survive
comparison with simulated annealing on the same QUBO, so QuGrid ships a
competent, seeded implementation (Metropolis sweeps, geometric temperature
schedule, restarts). It is also the practical workhorse for instances beyond
statevector reach (n > ~20).

For hardware annealing, export the same problem via
:func:`qugrid.adapters.to_bqm` and submit through D-Wave Ocean.
"""

from __future__ import annotations

import numpy as np

from qugrid.problems.base import CombinatorialProblem
from qugrid.solvers.base import Result, attach_reference, finish_combinatorial, timed


def solve_sa(
    problem: CombinatorialProblem,
    n_sweeps: int = 1500,
    n_restarts: int = 8,
    t_hot: float | None = None,
    t_cold: float | None = None,
    seed: int = 0,
    **_ignored,
) -> Result:
    """Simulated annealing on the Ising form of ``problem.qubo``.

    Temperatures default to the problem's own energy scale: ``t_hot`` at the
    maximum possible single-flip move, ``t_cold`` three orders below.
    """
    ising = problem.qubo.to_ising()
    n = ising.n
    h = ising.h
    jsym = ising.j + ising.j.T  # Ising.j is strictly upper triangular by construction
    scale = float(np.abs(h).max() + np.abs(jsym).sum(axis=1).max()) or 1.0
    t_hot = t_hot if t_hot is not None else 2.0 * scale
    t_cold = t_cold if t_cold is not None else 1e-3 * scale
    temps = t_hot * (t_cold / t_hot) ** (np.arange(n_sweeps) / max(n_sweeps - 1, 1))

    rng = np.random.default_rng(seed)
    res = Result(solver="sa", problem=problem)
    best_e = np.inf
    best_s: np.ndarray | None = None
    with timed(res.resources):
        for _restart in range(n_restarts):
            s = rng.choice([-1.0, 1.0], size=n)
            local = jsym @ s + h  # d E / d s_i contribution
            e = float(ising.energy(s))
            for t in temps:
                for i in rng.permutation(n):
                    de = -2.0 * s[i] * local[i]
                    if de <= 0 or rng.random() < np.exp(-de / t):
                        s[i] = -s[i]
                        e += de
                        local += 2.0 * s[i] * jsym[:, i]
                if e < best_e - 1e-12:
                    best_e, best_s = e, s.copy()
            res.history.append(best_e)
    assert best_s is not None
    x = ((1.0 - best_s) / 2.0).astype(int)  # s = 1 - 2x
    finish_combinatorial(res, problem, x)
    res.resources.update(
        {"n_vars": n, "sweeps": n_sweeps, "restarts": n_restarts, "seed": seed}
    )
    return attach_reference(res, problem)
