"""Tabu search and parallel tempering: the stronger classical opponents.

The contract mirrors simulated annealing: seeded, deterministic, gap 0 on
every bundled small formulation in well under a second.
"""

from __future__ import annotations

import pytest

import qugrid as qg
from qugrid.classical.dispatch import GenParams

GENS = [
    GenParams("base", pmin=0, pmax=90, c2=0.02, c1=10, c0=50, startup=100),
    GenParams("peaker", pmin=0, pmax=60, c2=0.04, c1=20, c0=30, startup=80),
]


def _problems():
    return [
        qg.problems.Islanding(qg.cases.case9()),
        qg.problems.UnitCommitment(GENS, [60.0, 130.0], power_bits=2),
        qg.problems.PMUPlacement(qg.cases.case5()),
    ]


@pytest.mark.parametrize("solver", ["tabu", "pt"])
def test_gap_zero_on_bundled_problems_three_seeds(solver, recwarn):
    for problem in _problems():
        for seed in (0, 1, 2):
            res = qg.solve(problem, solver=solver, seed=seed)
            assert res.feasible
            assert res.gap() == pytest.approx(0.0, abs=1e-9)
            assert res.resources["wall_time_s"] < 10.0


@pytest.mark.parametrize("solver", ["tabu", "pt"])
def test_same_seed_reproduces_the_result(solver, recwarn):
    problem = qg.problems.Islanding(qg.cases.case9())
    a = qg.solve(problem, solver=solver, seed=3)
    b = qg.solve(problem, solver=solver, seed=3)
    assert a.objective == b.objective
    assert (a.x == b.x).all()


def test_registry_lists_the_new_solvers():
    prob = qg.problems.Islanding(qg.cases.case9())
    with pytest.raises(ValueError, match="tabu"):
        qg.solve(prob, solver="not-a-solver")
