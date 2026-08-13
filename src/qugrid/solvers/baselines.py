"""Stronger classical baselines: tabu search and parallel tempering.

Exhaustive enumeration and plain simulated annealing are weak opponents for
a quantum claim. Parallel tempering is the strongest widely reported
classical heuristic on sparse QUBOs, and tabu search is the standard
deterministic local-search opponent — both appear as the classical side of
published annealer comparisons on power system problems (Kaseb et al.,
"Quantum annealing versus classical solvers for grid partitioning",
arXiv:2505.15978, D-Wave and Fujitsu digital annealer studies). Reviewers
of any study built on QuGrid will ask for them, so they ship with the same
seeded :class:`~qugrid.solvers.base.Result` contract as every other solver:

>>> qg.solve(prob, solver="tabu", seed=0)
>>> qg.solve(prob, solver="pt", seed=0)

Both operate on the Ising form with incremental local-field updates, the
same machinery as the built-in simulated annealing.
"""

from __future__ import annotations

import numpy as np

from qugrid.problems.base import CombinatorialProblem
from qugrid.solvers.base import Result, attach_reference, finish_combinatorial, timed


def solve_tabu(
    problem: CombinatorialProblem,
    n_iters: int = 2000,
    tenure: int | None = None,
    n_restarts: int = 4,
    seed: int = 0,
    **_ignored,
) -> Result:
    """Tabu search: best non-tabu single flip, aspiration on new global best.

    Each iteration flips the variable with the lowest resulting energy among
    those not flipped within the last ``tenure`` iterations; a tabu flip is
    allowed only when it beats the best energy seen (the aspiration
    criterion). ``tenure`` defaults to the common heuristic ``7 + n // 10``,
    capped at ``n // 2`` so small problems keep enough non-tabu moves to
    search with.
    """
    ising = problem.qubo.to_ising()
    n = ising.n
    jsym = ising.j + ising.j.T
    tenure = tenure if tenure is not None else max(2, min(7 + n // 10, n // 2))
    rng = np.random.default_rng(seed)

    res = Result(solver="tabu", problem=problem)
    best_e, best_s = np.inf, None
    with timed(res.resources):
        for _restart in range(n_restarts):
            s = rng.choice([-1.0, 1.0], size=n)
            local = jsym @ s + ising.h
            e = float(ising.energy(s))
            expiry = np.zeros(n, dtype=int)  # iteration until which a flip is tabu
            for it in range(n_iters):
                de = -2.0 * s * local
                allowed = expiry <= it
                # aspiration: a tabu move that makes a new global best is allowed
                allowed |= (e + de) < best_e - 1e-12
                if not allowed.any():
                    break
                cand = np.where(allowed, de, np.inf)
                i = int(np.argmin(cand))
                s[i] = -s[i]
                e += float(de[i])
                local += 2.0 * s[i] * jsym[:, i]
                expiry[i] = it + 1 + tenure
                if e < best_e - 1e-12:
                    best_e, best_s = e, s.copy()
            res.history.append(best_e)
    assert best_s is not None
    x = ((1.0 - best_s) / 2.0).astype(int)
    finish_combinatorial(res, problem, x)
    res.resources.update(
        {"n_vars": n, "iters": n_iters, "tenure": tenure, "restarts": n_restarts, "seed": seed}
    )
    return attach_reference(res, problem)


def solve_pt(
    problem: CombinatorialProblem,
    n_rounds: int = 300,
    n_replicas: int = 10,
    t_hot: float | None = None,
    t_cold: float | None = None,
    seed: int = 0,
    **_ignored,
) -> Result:
    """Parallel tempering: replicas on a temperature ladder with swap moves.

    Each round performs one Metropolis sweep per replica, then attempts a
    swap between every adjacent temperature pair with the standard exchange
    probability ``min(1, exp((1/T_i - 1/T_j)(E_i - E_j)))``. Hot replicas
    cross barriers, cold replicas refine, and exchanges move good
    configurations down the ladder — the mechanism that makes tempering the
    reported strongest classical heuristic on sparse QUBOs. Temperatures
    span the problem's own energy scale geometrically, like the built-in
    simulated annealing.
    """
    ising = problem.qubo.to_ising()
    n = ising.n
    h = ising.h
    jsym = ising.j + ising.j.T
    scale = float(np.abs(h).max() + np.abs(jsym).sum(axis=1).max()) or 1.0
    t_hot = t_hot if t_hot is not None else 2.0 * scale
    t_cold = t_cold if t_cold is not None else 1e-3 * scale
    temps = t_hot * (t_cold / t_hot) ** (np.arange(n_replicas) / max(n_replicas - 1, 1))
    betas = 1.0 / temps

    rng = np.random.default_rng(seed)
    res = Result(solver="pt", problem=problem)
    with timed(res.resources):
        spins = rng.choice([-1.0, 1.0], size=(n_replicas, n))
        locals_ = spins @ jsym + h  # row r: local field of replica r
        energies = np.array([float(ising.energy(s)) for s in spins])
        best_e = float(energies.min())
        best_s = spins[int(np.argmin(energies))].copy()
        swaps_accepted = 0
        for _round in range(n_rounds):
            for r in range(n_replicas):
                s, local = spins[r], locals_[r]
                e, t = energies[r], temps[r]
                for i in rng.permutation(n):
                    de = -2.0 * s[i] * local[i]
                    if de <= 0 or rng.random() < np.exp(-de / t):
                        s[i] = -s[i]
                        e += de
                        local += 2.0 * s[i] * jsym[:, i]
                energies[r] = e
                if e < best_e - 1e-12:
                    best_e, best_s = float(e), s.copy()
            for r in range(n_replicas - 1):  # adjacent swaps, hot to cold
                accept = (betas[r] - betas[r + 1]) * (energies[r] - energies[r + 1])
                if accept >= 0 or rng.random() < np.exp(accept):
                    spins[[r, r + 1]] = spins[[r + 1, r]]
                    locals_[[r, r + 1]] = locals_[[r + 1, r]]
                    energies[[r, r + 1]] = energies[[r + 1, r]]
                    swaps_accepted += 1
            res.history.append(best_e)
    x = ((1.0 - best_s) / 2.0).astype(int)
    finish_combinatorial(res, problem, x)
    res.resources.update(
        {
            "n_vars": n,
            "rounds": n_rounds,
            "replicas": n_replicas,
            "swaps_accepted": swaps_accepted,
            "seed": seed,
        }
    )
    return attach_reference(res, problem)
