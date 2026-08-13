"""QAOA variants from the literature, measured on power system problems.

WHAT
    Three published QAOA modifications, each reproduced with QuGrid's exact
    statevector solver on a bundled problem, so every claim is a measured
    probability, not a citation:

      * Warm-start QAOA — Egger, Marecek, Woerner, Quantum 5, 479 (2021):
        start from a product state biased toward a classical solution and
        mix around it. Applied to unit commitment by Salgado, Sequeira,
        Santos (arXiv:2412.11312). Here: controlled islanding of the WSCC
        9-bus system, warm-started from the simulated-annealing answer —
        the pipeline a practitioner would actually run.
      * XY (one-hot) mixer — Wang, Hadfield, Jiang, Rieffel, Phys. Rev. A
        101, 012320 (2020): exact ring-XY evolution conserves Hamming
        weight inside each one-hot group, so one-hot constraints hold with
        probability 1 and need no penalty. Run on IBM hardware for
        single-period unit commitment by Mohseni et al. (arXiv:2603.00260).
        Here: a 2-unit, 3-level one-hot dispatch.
      * Fixed transferred angles — Jing, Wang, Li, Communications
        Engineering 2, 12 (2023): angles optimized on one instance are
        reused on another, skipping the classical optimization loop
        entirely. Here: depth-1 angles from 9-bus islanding transferred to
        14-bus islanding.

WHY QUANTUM
    These are the standard answers to QAOA's two practical failures — flat
    starting distributions and expensive angle optimization. Each variant
    is exact statevector mathematics here; export via qugrid.adapters to
    add shots, noise, or hardware.

EXPECTED OUTPUT
    Three measured comparisons and a validation block. On the bundled
    instances:

      * warm start: depth-1 success probability on 9-bus islanding rises
        from 0.009 (vanilla, uniform start) to 0.128 when warm-started at
        the simulated-annealing solution — one order of magnitude from the
        same circuit depth.
      * XY mixer: on the one-hot dispatch toy, 100.0% of the output
        distribution satisfies the one-hot constraints and the optimum is
        hit exactly; the penalty encoding of the same problem puts 22% of
        its mass on feasible states.
      * angle transfer: 9-bus angles land within 5% of the expectation the
        14-bus optimizer reaches after ~900 circuit evaluations, at the
        cost of one evaluation; arbitrary angles miss by ~90%.

    Runtime is about 10 seconds.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

import qugrid as qg
from qugrid.methods import ConstrainedQUBOProblem
from qugrid.problems.builder import QUBOBuilder
from qugrid.solvers import XYMixer


def one_hot_dispatch(penalty_onehot: float) -> ConstrainedQUBOProblem:
    """2 units x 3 one-hot output levels; balance folded into the objective."""
    levels = [(10.0, 20.0, 30.0), (15.0, 25.0, 35.0)]
    costs = [(12.0, 25.0, 41.0), (14.0, 22.0, 38.0)]
    bld = QUBOBuilder()
    xs = [[bld.var(f"x[{u},{level}]") for level in range(3)] for u in range(2)]
    for u in range(2):
        for level in range(3):
            bld.add_linear(xs[u][level], costs[u][level])
    terms = [(xs[u][level], levels[u][level]) for u in range(2) for level in range(3)]
    bld.add_squared_penalty(terms, -55.0, weight=0.5)
    if penalty_onehot > 0:
        for u in range(2):
            bld.add_squared_penalty(
                [(xs[u][level], 1.0) for level in range(3)], -1.0, weight=penalty_onehot
            )
    one_hot = [([(3 * u + level, 1.0) for level in range(3)], -1.0) for u in range(2)]
    return ConstrainedQUBOProblem(bld.build(), one_hot)


def feasible_mass(res, problem) -> float:
    return sum(
        p
        for bits, p, _ in res.top_states
        if problem.is_feasible(np.array([int(c) for c in bits]))
    )


def main() -> int:
    warnings.simplefilter("ignore", RuntimeWarning)

    # ------------------------------------------------------------- warm start
    isl9 = qg.problems.Islanding(qg.cases.case9())
    classical = qg.solve(isl9, solver="sa", seed=0)  # the cheap warm starter
    vanilla = qg.solve(isl9, solver="qaoa", p=1, seed=0)
    warm = qg.solve(isl9, solver="qaoa", p=1, seed=0, warm_start=classical.x.astype(float))

    print("Warm-start QAOA (Egger 2021), islanding case9, depth p=1")
    print(f"  vanilla    P(optimum) = {vanilla.success_probability():.4f}")
    print(f"  warm@SA    P(optimum) = {warm.success_probability():.4f}   "
          f"(start: simulated-annealing answer, epsilon=0.25)")

    # -------------------------------------------------------------- XY mixer
    pure = one_hot_dispatch(penalty_onehot=0.0)
    penalized = one_hot_dispatch(penalty_onehot=50.0)
    xy = qg.solve(pure, solver="qaoa", p=2, seed=0, mixer=XYMixer([[0, 1, 2], [3, 4, 5]]))
    pen = qg.solve(penalized, solver="qaoa", p=2, seed=0)

    print("\nXY mixer (Wang 2020), 2-unit 3-level one-hot dispatch, p=2")
    print(f"  XY mixer   P(one-hot satisfied) = {feasible_mass(xy, pure):.4f}, "
          f"gap = {100 * xy.gap():.2f}%  (no penalty term at all)")
    print(f"  penalty    P(one-hot satisfied) = {feasible_mass(pen, penalized):.4f}")

    # --------------------------------------------------------- angle transfer
    isl14 = qg.problems.Islanding(qg.cases.case14())
    donor = qg.solve(isl9, solver="qaoa", p=1, seed=0)
    angles = (donor.resources["gammas"], donor.resources["betas"])
    optimized = qg.solve(isl14, solver="qaoa", p=1, seed=0)
    transferred = qg.solve(isl14, solver="qaoa", p=1, seed=0, angles=angles)
    arbitrary = qg.solve(isl14, solver="qaoa", p=1, seed=0, angles=([0.05], [2.9]))

    e_opt = optimized.resources["expectation"]
    e_tra = transferred.resources["expectation"]
    e_arb = arbitrary.resources["expectation"]
    print("\nFixed transferred angles (Jing 2023), islanding case9 -> case14, p=1")
    print(f"  optimized on case14    <H> = {e_opt:8.2f}   "
          f"({optimized.resources['evaluations']} circuit evaluations)")
    print(f"  transferred from case9 <H> = {e_tra:8.2f}   (1 evaluation)")
    print(f"  arbitrary angles       <H> = {e_arb:8.2f}   (1 evaluation)")

    # --------------------------------------------------------------- validation
    assert warm.success_probability() > 10 * vanilla.success_probability()
    assert feasible_mass(xy, pure) > 1.0 - 1e-9
    assert abs(xy.gap()) < 1e-9 and xy.feasible
    assert feasible_mass(pen, penalized) < 0.999
    assert abs(e_tra - e_opt) / abs(e_opt) < 0.10, "transfer off by more than 10%"
    assert e_tra < e_arb, "transferred angles no better than arbitrary ones"
    assert transferred.resources["evaluations"] == 1

    print("\nVALIDATION PASSED: warm start lifts depth-1 success probability by >10x;")
    print("the XY mixer holds one-hot feasibility at 1 with gap 0 where penalties")
    print("leak; case9 angles land within 10% of the case14 optimum with a single")
    print("circuit evaluation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
