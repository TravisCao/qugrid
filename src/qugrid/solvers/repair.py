"""Greedy repair of infeasible samples, and the ``random+repair`` baseline.

Sampling solvers put probability mass on infeasible bitstrings whenever
penalty terms compete with cost terms. REGRID-QAOA (Jiang et al.,
arXiv:2606.15083, islanding on IBM hardware at 9-57 buses) showed that a
cheap classical repair of every measured sample recovers most of that mass.
The honest control is to give plain random sampling the same repair —
Gaidai and Mukherjee (arXiv:2607.15543) report hybrid quantum pipelines at
5-13 qubits failing to beat exactly that baseline — so QuGrid ships both:

>>> qg.solve(prob, solver="qaoa", repair="greedy")  # repaired quantum
>>> qg.solve(prob, solver="random+repair")          # repaired cheap baseline

Repair descends :meth:`~qugrid.problems.base.CombinatorialProblem.
constraint_residual`, so it is only as sharp as that measure: formulations
with a graded residual (unit commitment, PMU placement) repair step by step;
the default 0/1 indicator limits repair to single-flip fixes.
"""

from __future__ import annotations

import numpy as np

from qugrid.problems.base import CombinatorialProblem
from qugrid.solvers.base import Result, attach_reference, finish_combinatorial, timed


def greedy_repair(
    problem: CombinatorialProblem, x: np.ndarray, max_flips: int | None = None
) -> np.ndarray:
    """Flip the bits that most reduce ``constraint_residual`` until feasible.

    Each accepted move — a single flip, or the best pair flip when every
    single flip stalls — strictly reduces ``(residual, QUBO energy)`` in
    lexicographic order: residual first, and on residual plateaus the move
    that lowers the penalized energy most (penalties point toward
    feasibility even where the residual is locally flat — a unit commitment
    period with everything off moves no residual with any single flip, but
    the balance penalty walks it toward the demand). The strict decrease of
    the pair over a finite state space rules out cycles, so the loop
    terminates; when no move decreases the pair, the best-effort iterate
    comes back and may still be infeasible — check ``problem.is_feasible``
    on the output. Feasible input returns unchanged.
    """
    x = np.asarray(x, dtype=int).copy()
    if problem.is_feasible(x):
        return x
    qubo = problem.qubo
    if max_flips is None:
        max_flips = 4 * qubo.n

    def better_move(r: float, e: float) -> tuple[float, float, tuple[int, ...]] | None:
        """Best single flip decreasing (residual, energy); pair flips at stalls.

        Pair moves cover the exchanges a binary power expansion needs (bit
        pattern 10 -> 01 is a +/- step no single flip performs); they only
        run when every single flip fails, so the quadratic scan stays rare.
        """
        for flips in ([(i,) for i in range(qubo.n)], None):
            if flips is None:
                flips = [(i, j) for i in range(qubo.n) for j in range(i + 1, qubo.n)]
            best: tuple[float, float, tuple[int, ...]] | None = None
            for move in flips:
                for i in move:
                    x[i] ^= 1
                ri = problem.constraint_residual(x)
                if ri < r - 1e-12:
                    key = (ri, float(qubo.energy(x)), move)
                elif ri < r + 1e-12:  # residual plateau: descend penalized energy
                    ei = float(qubo.energy(x))
                    key = (r, ei, move) if ei < e - 1e-12 else None
                else:
                    key = None
                if key is not None and (best is None or key[:2] < best[:2]):
                    best = key
                for i in move:
                    x[i] ^= 1
            if best is not None:
                return best
        return None

    r = problem.constraint_residual(x)
    e = float(qubo.energy(x))
    for _ in range(max_flips):
        move = better_move(r, e)
        if move is None:
            break
        r, e = move[0], move[1]
        for i in move[2]:
            x[i] ^= 1
        if problem.is_feasible(x):
            break
    return x


def repair_result(res: Result, problem: CombinatorialProblem) -> Result:
    """Repair a solver's answer and, when present, its whole output distribution.

    Replaces ``res.x`` with the best feasible bitstring among the repaired
    candidates (kept as-is when nothing feasible is reachable). For solvers
    that report ``top_states``, also computes ``P(optimum | repaired)`` — the
    probability mass whose repaired bitstring lands on the reference optimum
    — into ``res.extras`` alongside the raw ``P(optimum)``.
    """
    candidates = [greedy_repair(problem, res.x)]
    if res.top_states:
        ref = (res.reference or {}).get("objective")
        p_raw = res.success_probability()
        p_rep = 0.0
        for bitstring, prob, _energy in res.top_states:
            xs = np.array([int(c) for c in bitstring], dtype=int)
            xr = greedy_repair(problem, xs)
            candidates.append(xr)
            if ref is not None and problem.is_feasible(xr):
                if abs(float(problem.qubo.energy(xr)) - ref) <= 1e-9 * max(1.0, abs(ref)):
                    p_rep += prob
        if ref is not None and p_raw is not None:
            res.extras["p_optimum_raw"] = float(p_raw)
            res.extras["p_optimum_repaired"] = float(p_rep)
    feasible = [c for c in candidates if problem.is_feasible(c)]
    if feasible:
        best = min(feasible, key=lambda c: float(problem.qubo.energy(c)))
        finish_combinatorial(res, problem, best)
    res.resources["repair"] = "greedy"
    return res


def solve_random_repair(
    problem: CombinatorialProblem, seed: int = 0, samples: int = 1000, **_ignored
) -> Result:
    """Uniform random sampling plus greedy repair of every unique sample.

    The cheap baseline any repaired quantum pipeline must beat: no circuit,
    no annealer, just ``samples`` uniform bitstrings pushed through the same
    repair. Returns the best feasible repaired sample (best-energy sample
    when nothing repairs to feasible).
    """
    rng = np.random.default_rng(seed)
    res = Result(solver="random+repair", problem=problem)
    with timed(res.resources):
        n = problem.qubo.n
        xs = np.unique(rng.integers(0, 2, size=(samples, n)), axis=0)
        repaired = np.array([greedy_repair(problem, x) for x in xs])
        energies = problem.qubo.energy(repaired)
        feas = np.array([problem.is_feasible(x) for x in repaired])
        pool = np.flatnonzero(feas) if feas.any() else np.arange(len(repaired))
        best = pool[int(np.argmin(energies[pool]))]
    finish_combinatorial(res, problem, repaired[best])
    res.resources.update(
        {"samples": samples, "unique_samples": len(xs), "repair": "greedy", "seed": seed}
    )
    return attach_reference(res, problem)
