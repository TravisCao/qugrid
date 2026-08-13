"""Slack-free constraint handling: unbalanced penalties, the augmented
Lagrangian loop, and the dynamic-range diagnostic.

The equivalence standard throughout: enumerate every bitstring and compare
the *decoded, feasibility-checked* optimum across encodings — never trust
one encoding to referee another.
"""

from __future__ import annotations

import numpy as np
import pytest

import qugrid as qg
from qugrid.classical.dispatch import GenParams
from qugrid.methods import AugmentedLagrangianLoop, ConstrainedQUBOProblem
from qugrid.problems.base import QUBO
from qugrid.problems.builder import QUBOBuilder
from qugrid.problems.unit_commitment import UnitCommitment

GENS = [
    GenParams("base", pmin=0, pmax=90, c2=0.02, c1=10, c0=50, startup=100),
    GenParams("peaker", pmin=0, pmax=60, c2=0.04, c1=20, c0=30, startup=80),
]
DEMAND = [60.0, 130.0]


def _all_bits(n: int) -> np.ndarray:
    states = np.arange(2**n, dtype=np.int64)
    return ((states[:, None] >> np.arange(n)) & 1).astype(float)


def _argmin(qubo: QUBO) -> np.ndarray:
    bits = _all_bits(qubo.n)
    return bits[int(np.argmin(qubo.energy(bits)))].astype(int)


# ------------------------------------------------- unbalanced vs slack encoding


def _pmu_qubo(net, method: str, lam=None) -> QUBO:
    """Minimum PMU placement (dominating set) built through add_inequality."""
    a = net.adjacency()
    n = net.n_bus
    bld = QUBOBuilder()
    xs = [bld.var(f"x{i}") for i in range(n)]
    for i in xs:
        bld.add_linear(i, 1.0)
    for i in range(n):
        terms = [(xs[i], 1.0)] + [(xs[j], 1.0) for j in np.flatnonzero(a[i])]
        bld.add_inequality(terms, -1.0, weight=10.0, method=method, lam=lam)
    return bld.build()


def _covers(net, bits: np.ndarray) -> bool:
    a = net.adjacency()
    placed = bits[: net.n_bus].astype(bool)
    covered = placed.copy()
    for i in np.flatnonzero(placed):
        covered[np.flatnonzero(a[i])] = True
    return bool(covered.all())


def test_unbalanced_reproduces_slack_optimum_on_pmu_case5():
    net = qg.cases.case5()
    slack = _pmu_qubo(net, "slack")
    unbal = _pmu_qubo(net, "unbalanced", lam=(1.0, 1.0))

    x_slack = _argmin(slack)
    x_unbal = _argmin(unbal)
    assert _covers(net, x_slack) and _covers(net, x_unbal)

    reference = qg.problems.PMUPlacement(net).reference()["n_pmu"]
    assert int(x_slack[: net.n_bus].sum()) == reference
    assert int(x_unbal.sum()) == reference

    # The selling point: the same constraints with zero slack bits.
    assert unbal.n == net.n_bus
    assert slack.n > unbal.n


def test_unbalanced_reproduces_slack_optimum_on_toy_constraint():
    # min x0 + 2 x1 + 3 x2  s.t.  x0 + x1 + x2 >= 2, both encodings.
    def build(method, lam=None):
        bld = QUBOBuilder()
        xs = [bld.var(f"x{i}") for i in range(3)]
        for i, c in zip(xs, (1.0, 2.0, 3.0)):
            bld.add_linear(i, c)
        bld.add_inequality([(i, 1.0) for i in xs], -2.0, weight=8.0, method=method, lam=lam)
        return bld.build()

    x_slack = _argmin(build("slack"))[:3]
    x_unbal = _argmin(build("unbalanced", lam=(2.0, 4.0)))
    assert x_slack.sum() >= 2 and x_unbal.sum() >= 2
    assert np.array_equal(x_slack, [1, 1, 0])
    assert np.array_equal(x_unbal, [1, 1, 0])


def test_slack_encoding_rejects_non_integer_left_side():
    bld = QUBOBuilder()
    i = bld.var("x0")
    with pytest.raises(ValueError, match="integer-valued"):
        bld.add_inequality([(i, 0.5)], 0.0)


def test_slack_bits_are_unique_per_constraint():
    bld = QUBOBuilder()
    xs = [bld.var(f"x{i}") for i in range(2)]
    bld.add_inequality([(xs[0], 1.0)], 0.0, weight=1.0)
    bld.add_inequality([(xs[1], 1.0)], 0.0, weight=1.0)
    assert len(set(bld._names)) == bld.n  # no silent slack-bit sharing


# ------------------------------------------------------ augmented Lagrangian


def _uc_pure_and_constraints():
    """UC with the balance penalty stripped, plus the balance equalities."""
    uc0 = UnitCommitment(GENS, DEMAND, power_bits=2, weight_balance=0.0)
    idx = {name: i for i, name in enumerate(uc0.qubo.names)}
    constraints = []
    for t in range(len(DEMAND)):
        terms = []
        for g, gen in enumerate(GENS):
            terms.append((idx[f"u[{g},{t}]"], float(gen.pmin)))
            for k in range(uc0.power_bits):
                terms.append((idx[f"b[{g},{t},{k}]"], float(uc0.delta[g] * 2**k)))
        constraints.append((terms, -float(DEMAND[t])))
    return uc0, constraints


def test_al_loop_reaches_uc_feasible_optimum_within_ten_iterations():
    uc0, constraints = _uc_pure_and_constraints()
    target = qg.solve(UnitCommitment(GENS, DEMAND, power_bits=2), solver="exact")
    # The penalized optimum has zero residual, so its energy is the pure cost.
    loop = AugmentedLagrangianLoop(
        uc0.qubo, constraints, solver="sa", lam0=1.0, alpha=2.0, iters=10
    )
    res = loop.run(seed=0)
    assert res.feasible
    assert res.objective == pytest.approx(target.objective, abs=1e-6)
    assert res.gap() == pytest.approx(0.0, abs=1e-9)
    assert len(loop.history) <= 10
    assert res.resources["iterations"] == len(loop.history)
    # The whole point: the loop never builds the stretched fixed-penalty QUBO.
    assert uc0.qubo.dynamic_range() < 1e3


def test_al_loop_reports_infeasibility_honestly():
    bld = QUBOBuilder()
    xs = [bld.var(f"x{i}") for i in range(2)]
    for i in xs:
        bld.add_linear(i, 1.0)
    impossible = [([(i, 1.0) for i in xs], 5.0)]  # x0 + x1 + 5 = 0
    loop = AugmentedLagrangianLoop(bld.build(), impossible, solver="sa", iters=3)
    res = loop.run(seed=0)
    assert not res.feasible
    assert len(loop.history) == 3  # no early exit without a feasible iterate


def test_constrained_problem_reference_enumerates_feasible_set():
    uc0, constraints = _uc_pure_and_constraints()
    prob = ConstrainedQUBOProblem(uc0.qubo, constraints)
    ref = prob.reference()
    assert prob.is_feasible(ref["x"])
    assert ref["objective"] == pytest.approx(2908.0, abs=1e-6)


# ------------------------------------------------------------- dynamic range


def test_dynamic_range_exact_on_hand_built_qubo():
    qubo = QUBO(q=np.array([[1.0, 25.0], [25.0, 100.0]]))
    # coefficients: linear {1, 100}, coupling 2*25 = 50 -> ratio 100
    assert qubo.dynamic_range() == pytest.approx(100.0)
    assert qubo.dynamic_range(db=True) == pytest.approx(40.0)


def test_dynamic_range_monotone_under_penalty_rescale():
    low = UnitCommitment(GENS, DEMAND, power_bits=2, weight_balance=1e3)
    high = UnitCommitment(GENS, DEMAND, power_bits=2, weight_balance=1e4)
    assert high.qubo.dynamic_range() >= low.qubo.dynamic_range()


def test_solve_warns_on_stretched_qubo_and_stays_quiet_otherwise():
    uc = UnitCommitment(GENS, DEMAND, power_bits=2)  # measured range 3.9e3
    with pytest.warns(RuntimeWarning, match="dynamic range"):
        qg.solve(uc, solver="sa", seed=0)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)
        qg.solve(uc, solver="exact")  # exact enumeration is immune: no warning
        qg.solve(qg.problems.Islanding(qg.cases.case9()), solver="sa", seed=0)
