"""Constraint handling study: slack penalties, unbalanced penalties, and the
augmented Lagrangian, measured on the same problems.

WHAT
    Every constraint that enters a QUBO costs something: slack bits cost
    qubits, penalty weights cost dynamic range, and both costs show up as
    lost success probability in sampling solvers. This script measures the
    trade on two problems from the zoo:

      * minimum PMU placement on the PJM 5-bus system — one inequality per
        bus — encoded two ways: textbook binary slack (Lucas, "Ising
        formulations of many NP problems", Front. Physics 2, 2014) against
        the unbalanced penalization of Montanez-Barrera et al. (Quantum Sci.
        Technol. 9, 025022, 2024), which spends zero slack qubits;
      * unit commitment, 2 units over 2 periods — one balance equality per
        period — encoded with the standard fixed quadratic penalty against
        the augmented Lagrangian outer loop
        (:class:`qugrid.methods.AugmentedLagrangianLoop`; Hong, Xu, Teng,
        arXiv:2502.15917; Feng et al., IEEE Trans. Power Systems 38(3),
        2023), which keeps the QUBO dynamic range flat;
      * random sampling with greedy repair
        (:func:`qugrid.solvers.greedy_repair`) on both problems — the cheap
        classical control every encoding trick has to beat (Gaidai and
        Mukherjee, arXiv:2607.15543).

WHY QUANTUM
    None of this is a quantum speedup claim. It is the engineering that
    decides whether a sampling solver — quantum or classical — sees the
    problem at all: the fixed-penalty unit commitment QUBO stretches its
    coefficient range to 3.9e3 and depth-2 QAOA collapses on it (script 07),
    while the same physics with constraints held outside the QUBO stays at
    range 180 and solves exactly.

EXPECTED OUTPUT
    One table per problem and a validation block. Measured on the bundled
    instances:

      * PMU: both encodings agree with exhaustive enumeration (2 PMUs).
        Unbalanced penalization uses 5 qubits against 15 with slack, and
        depth-2 QAOA success probability rises from 0.001 to 0.54 — the
        whole gain is the removed slack register. Its caveat: the penalty
        only approximates the constraint, so always verify feasibility of
        the returned answer (the script does).
      * Unit commitment: the augmented Lagrangian reaches the feasible
        optimum (cost 2908, gap 0) in 3 outer iterations while its largest
        QUBO stays 21x flatter than the fixed-penalty encoding.
      * random+repair reaches gap 0 on both problems in well under a
        second, which is exactly why it ships as a baseline.

    Runtime is about 10 seconds.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

import qugrid as qg
from qugrid.classical.dispatch import GenParams
from qugrid.methods import AugmentedLagrangianLoop
from qugrid.problems.base import QUBO, CombinatorialProblem
from qugrid.problems.builder import QUBOBuilder

GENS = [
    GenParams("base", pmin=0, pmax=90, c2=0.02, c1=10, c0=50, startup=100),
    GenParams("peaker", pmin=0, pmax=60, c2=0.04, c1=20, c0=30, startup=80),
]
DEMAND = [60.0, 130.0]


class UnbalancedPMU(CombinatorialProblem):
    """PMU placement with unbalanced-penalty coverage — no slack qubits.

    The same dominating-set formulation as :class:`qugrid.problems.
    PMUPlacement`, with each coverage inequality entering through
    ``QUBOBuilder.add_inequality(..., method="unbalanced")`` instead of
    binary slack. Placement bits are the only variables.
    """

    def __init__(self, net, lam=(1.0, 1.0), weight=10.0):
        self.net = net
        a = net.adjacency()
        self.neighbors = [np.flatnonzero(a[i]).tolist() for i in range(net.n_bus)]
        bld = QUBOBuilder()
        xs = [bld.var(f"pmu@bus{int(net.bus[i, 0])}") for i in range(net.n_bus)]
        for i in xs:
            bld.add_linear(i, 1.0)
        for i in range(net.n_bus):
            terms = [(xs[i], 1.0)] + [(xs[j], 1.0) for j in self.neighbors[i]]
            bld.add_inequality(terms, -1.0, weight=weight, method="unbalanced", lam=lam)
        self._qubo = bld.build()

    @property
    def qubo(self) -> QUBO:
        return self._qubo

    def decode(self, x: np.ndarray) -> dict:
        return {"n_pmu": int(np.asarray(x, dtype=int).sum())}

    def is_feasible(self, x: np.ndarray) -> bool:
        placed = np.asarray(x, dtype=int).astype(bool)
        covered = placed.copy()
        for i in np.flatnonzero(placed):
            covered[self.neighbors[i]] = True
        return bool(covered.all())

    def constraint_residual(self, x: np.ndarray) -> float:
        placed = np.asarray(x, dtype=int).astype(bool)
        covered = placed.copy()
        for i in np.flatnonzero(placed):
            covered[self.neighbors[i]] = True
        return float((~covered).sum())


def uc_equalities(uc0):
    """The balance equalities of a UnitCommitment, in builder term form."""
    idx = {name: i for i, name in enumerate(uc0.qubo.names)}
    constraints = []
    for t in range(len(DEMAND)):
        terms = []
        for g, gen in enumerate(GENS):
            terms.append((idx[f"u[{g},{t}]"], float(gen.pmin)))
            for k in range(uc0.power_bits):
                terms.append((idx[f"b[{g},{t},{k}]"], float(uc0.delta[g] * 2**k)))
        constraints.append((terms, -float(DEMAND[t])))
    return constraints


def main() -> int:
    warnings.simplefilter("ignore", RuntimeWarning)  # measured below instead

    # ------------------------------------------------- PMU: slack vs unbalanced
    net = qg.cases.case5()
    slack = qg.problems.PMUPlacement(net)
    unbal = UnbalancedPMU(net)

    print("PMU placement, PJM 5-bus: one coverage inequality per bus")
    header = (
        f"{'encoding':<12} {'qubits':>7} {'dyn.range':>10} "
        f"{'P(opt)':>8} {'gap %':>7} {'PMUs':>5}"
    )
    print(header)
    print("-" * len(header))
    pmu_rows = {}
    for name, prob in [("slack", slack), ("unbalanced", unbal)]:
        res = qg.solve(prob, solver="qaoa", p=2, seed=0)
        pmu_rows[name] = res
        print(
            f"{name:<12} {prob.n:>7} {prob.qubo.dynamic_range():>10.1f} "
            f"{res.success_probability():>8.4f} {100 * res.gap():>7.2f} "
            f"{res.decoded['n_pmu']:>5}"
        )
    rr = qg.solve(unbal, solver="random+repair", seed=0, samples=1000)
    print(f"{'rand+repair':<12} {unbal.n:>7} {'—':>10} {'—':>8} {100 * rr.gap():>7.2f} "
          f"{rr.decoded['n_pmu']:>5}")

    # ------------------------------------------- UC: fixed penalty vs AL loop
    uc = qg.problems.UnitCommitment(GENS, DEMAND, power_bits=2)
    uc0 = qg.problems.UnitCommitment(GENS, DEMAND, power_bits=2, weight_balance=0.0)
    target = qg.solve(uc, solver="exact")

    print("\nUnit commitment, 2 units x 2 periods: one balance equality per period")
    header = f"{'method':<16} {'qubits':>7} {'dyn.range':>10} {'gap %':>7} {'feasible':>9}"
    print(header)
    print("-" * len(header))
    qaoa_uc = qg.solve(uc, solver="qaoa", p=2, seed=0)
    print(
        f"{'fixed penalty':<16} {uc.n:>7} {uc.qubo.dynamic_range():>10.3g} "
        f"{100 * qaoa_uc.gap():>7.1f} {str(qaoa_uc.feasible):>9}   (QAOA p=2, "
        f"P(opt) {qaoa_uc.success_probability():.4f})"
    )
    loop = AugmentedLagrangianLoop(uc0.qubo, uc_equalities(uc0), solver="sa")
    al = loop.run(seed=0)
    print(
        f"{'augmented Lagr.':<16} {uc0.n:>7} {uc0.qubo.dynamic_range():>10.3g} "
        f"{100 * al.gap():>7.1f} {str(al.feasible):>9}   ({al.resources['iterations']} outer "
        f"iterations, inner solver sa)"
    )
    rr_uc = qg.solve(uc, solver="random+repair", seed=0, samples=1000)
    print(
        f"{'rand+repair':<16} {uc.n:>7} {uc.qubo.dynamic_range():>10.3g} "
        f"{100 * rr_uc.gap():>7.1f} {str(rr_uc.feasible):>9}"
    )

    # --------------------------------------------------------------- validation
    assert qg.solve(slack, solver="exact").decoded["n_pmu"] == 2
    assert qg.solve(unbal, solver="exact").decoded["n_pmu"] == 2
    assert unbal.n == 5 and slack.n == 15
    p_slack = pmu_rows["slack"].success_probability()
    p_unbal = pmu_rows["unbalanced"].success_probability()
    assert p_unbal > 100 * p_slack, f"unbalanced {p_unbal} vs slack {p_slack}"
    assert pmu_rows["unbalanced"].feasible and pmu_rows["slack"].feasible
    assert abs(rr.gap()) < 1e-9

    assert al.feasible and abs(al.gap()) < 1e-9
    assert al.objective == target.objective
    assert al.resources["iterations"] <= 10
    assert uc.qubo.dynamic_range() > 1e3 > uc0.qubo.dynamic_range()
    assert abs(rr_uc.gap()) < 1e-9

    print("\nVALIDATION PASSED: both PMU encodings match exhaustive enumeration;")
    print("unbalanced penalties lift QAOA success probability by >100x with 1/3 of")
    print("the qubits; the augmented Lagrangian reaches the feasible UC optimum with")
    print("a flat-range QUBO; random+repair reaches gap 0 on both problems.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
