"""Greedy repair and the random+repair baseline.

Repair claims are empirical, so the tests are seeded batches, not single
states: repair must reach feasibility from every uniform random state on
the three bundled formulations.
"""

from __future__ import annotations

import numpy as np
import pytest

import qugrid as qg
from qugrid.classical.dispatch import GenParams
from qugrid.solvers.repair import greedy_repair

GENS = [
    GenParams("base", pmin=0, pmax=90, c2=0.02, c1=10, c0=50, startup=100),
    GenParams("peaker", pmin=0, pmax=60, c2=0.04, c1=20, c0=30, startup=80),
]


@pytest.fixture
def uc():
    return qg.problems.UnitCommitment(GENS, [60.0, 130.0], power_bits=2)


def test_repair_reaches_feasibility_from_random_states(uc):
    rng = np.random.default_rng(0)
    for problem, n_states in [
        (uc, 200),
        (qg.problems.Islanding(qg.cases.case9()), 100),
        (qg.problems.PMUPlacement(qg.cases.case5()), 100),
    ]:
        xs = rng.integers(0, 2, size=(n_states, problem.qubo.n))
        assert all(problem.is_feasible(greedy_repair(problem, x)) for x in xs)


def test_repair_returns_feasible_input_unchanged(uc):
    ref = uc.reference()["x"]
    assert np.array_equal(greedy_repair(uc, ref), ref)


def test_uc_needs_the_pair_flip_neighborhood(uc):
    # A one-flip local minimum of the residual: period 1 sits 20 MW short and
    # only the exchange 10 -> 01 in the peaker's power bits (+20 MW) fixes it.
    stuck = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0])
    assert not uc.is_feasible(stuck)
    assert uc.is_feasible(greedy_repair(uc, stuck))


def test_repaired_success_probability_reported_and_not_below_raw(uc):
    with pytest.warns(RuntimeWarning):  # stretched QUBO warning still fires
        res = qg.solve(uc, solver="qaoa", p=2, seed=0, repair="greedy")
    p_raw = res.extras["p_optimum_raw"]
    p_rep = res.extras["p_optimum_repaired"]
    assert p_rep >= p_raw
    assert p_rep > 0.0  # measured 0.011 where the raw mass is 0.000
    assert res.feasible and res.gap() == pytest.approx(0.0, abs=1e-9)
    assert "P(opt | repaired)" in res.summary()
    assert res.resources["repair"] == "greedy"


def test_no_repair_means_no_repair_fields(uc):
    with pytest.warns(RuntimeWarning):
        res = qg.solve(uc, solver="qaoa", p=1, maxiter=40, seed=0)
    assert "p_optimum_repaired" not in res.extras
    assert "repair" not in res.resources


def test_random_repair_reaches_gap_zero_seeded(uc):
    isl = qg.problems.Islanding(qg.cases.case9())
    res = qg.solve(isl, solver="random+repair", seed=0, samples=1200)
    assert res.feasible and res.gap() == pytest.approx(0.0, abs=1e-9)
    with pytest.warns(RuntimeWarning):
        res_uc = qg.solve(uc, solver="random+repair", seed=0, samples=1000)
    assert res_uc.feasible and res_uc.gap() == pytest.approx(0.0, abs=1e-9)


def test_unknown_repair_name_raises(uc):
    with pytest.raises(ValueError, match="greedy"):
        qg.solve(uc, solver="exact", repair="magic")


def test_repair_rejected_for_linear_problems():
    prob = qg.problems.dc_power_flow(qg.cases.case9())
    with pytest.raises(ValueError, match="combinatorial"):
        qg.solve(prob, solver="numpy", repair="greedy")
