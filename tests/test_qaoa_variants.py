"""Warm-start QAOA and the XY (one-hot) mixer.

Papers under test: Egger, Marecek, Woerner, Quantum 5, 479 (2021) for the
warm start; Wang, Hadfield, Jiang, Rieffel, PRA 101, 012320 (2020) for XY
mixing. Every claim is checked against exact statevector quantities.
"""

from __future__ import annotations

import numpy as np
import pytest

import qugrid as qg
from qugrid.methods import ConstrainedQUBOProblem
from qugrid.problems.builder import QUBOBuilder
from qugrid.solvers.mixers import WarmStartMixer, XYMixer
from qugrid.solvers.statevector import apply_unitary


@pytest.fixture
def islanding():
    return qg.problems.Islanding(qg.cases.case9())


# ------------------------------------------------------------ infrastructure


def test_apply_unitary_matches_direct_permutation():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    out = apply_unitary(psi.copy(), 3, [1], x_gate)
    assert np.allclose(out, psi[np.arange(8) ^ 2])  # flip bit 1 of the index


def test_warm_start_mixer_is_unitary_with_its_initial_state_as_eigenstate():
    ws = WarmStartMixer(np.array([0.3, 0.6, 0.45]), epsilon=0.25)
    psi0 = ws.initial_state(3)
    assert np.linalg.norm(psi0) == pytest.approx(1.0, abs=1e-12)
    moved = ws.apply(psi0.copy(), 3, 0.7)
    assert np.linalg.norm(moved) == pytest.approx(1.0, abs=1e-12)
    back = ws.apply(moved.copy(), 3, -0.7)
    assert np.allclose(back, psi0)
    # the warm-start state is a mixer ground state: invariant up to phase
    assert abs(np.vdot(ws.apply(psi0.copy(), 3, 0.9), psi0)) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------- warm start


def test_warm_start_at_optimum_beats_vanilla_at_depth_one(islanding):
    ref_x = islanding.reference()["x"].astype(float)
    vanilla = qg.solve(islanding, solver="qaoa", p=1, seed=0)
    warm = qg.solve(islanding, solver="qaoa", p=1, seed=0, warm_start=ref_x)
    # measured 0.128 vs 0.009: an order of magnitude at depth 1
    assert warm.success_probability() > vanilla.success_probability()
    assert warm.resources["mixer"] == "warm-start"


def test_epsilon_zero_freezes_the_warm_start_bits(islanding):
    ref_x = islanding.reference()["x"].astype(float)
    frozen = qg.solve(islanding, solver="qaoa", p=1, seed=0, warm_start=ref_x, epsilon=0.0)
    assert frozen.success_probability() == pytest.approx(1.0, abs=1e-9)


def test_relaxation_warm_start_runs(islanding):
    res = qg.solve(islanding, solver="qaoa", p=1, maxiter=60, seed=0, warm_start="relaxation")
    assert res.objective is not None
    assert res.resources["mixer"] == "warm-start"
    c = qg.solvers.relaxed_solution(islanding.qubo)
    assert c.shape == (islanding.n,)
    assert ((c >= 0.0) & (c <= 1.0)).all()


def test_warm_start_and_mixer_are_mutually_exclusive(islanding):
    with pytest.raises(ValueError, match="not both"):
        qg.solve(
            islanding,
            solver="qaoa",
            warm_start=np.full(islanding.n, 0.5),
            mixer=qg.solvers.XMixer(),
        )


# ----------------------------------------------------------------- XY mixer


def _dispatch_toy(penalty_onehot: float) -> ConstrainedQUBOProblem:
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


def test_xy_evolution_conserves_the_one_hot_subspace():
    xy = XYMixer([[0, 1, 2], [3, 4, 5]])
    psi = xy.apply(xy.initial_state(6), 6, 1.234)
    idx = np.arange(64)
    weight = [
        sum((idx >> q) & 1 for q in group) for group in ([0, 1, 2], [3, 4, 5])
    ]
    outside = ~((weight[0] == 1) & (weight[1] == 1))
    assert float((np.abs(psi) ** 2)[outside].sum()) < 1e-9


def test_xy_qaoa_certain_feasibility_where_penalties_leak():
    def feasible_mass(res, problem):
        return sum(
            p
            for bits, p, _ in res.top_states
            if problem.is_feasible(np.array([int(c) for c in bits]))
        )

    pure = _dispatch_toy(penalty_onehot=0.0)
    xy = qg.solve(pure, solver="qaoa", p=2, seed=0, mixer=XYMixer([[0, 1, 2], [3, 4, 5]]))
    assert xy.feasible
    assert xy.gap() == pytest.approx(0.0, abs=1e-9)
    assert feasible_mass(xy, pure) == pytest.approx(1.0, abs=1e-9)

    penalized = _dispatch_toy(penalty_onehot=50.0)
    vanilla = qg.solve(penalized, solver="qaoa", p=2, seed=0)
    assert feasible_mass(vanilla, penalized) < 0.999  # measured 0.221


def test_xy_string_requires_declared_groups(islanding):
    with pytest.raises(ValueError, match="one_hot_groups"):
        qg.solve(islanding, solver="qaoa", mixer="xy")


def test_xy_group_validation():
    with pytest.raises(ValueError, match="overlap"):
        XYMixer([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="at least 2"):
        XYMixer([[0]])


# -------------------------------------------------------------- fixed angles


def test_fixed_angles_reproduce_the_optimized_run(islanding):
    opt = qg.solve(islanding, solver="qaoa", p=2, seed=0)
    fixed = qg.solve(
        islanding,
        solver="qaoa",
        p=2,
        seed=0,
        angles=(opt.resources["gammas"], opt.resources["betas"]),
    )
    assert fixed.resources["expectation"] == pytest.approx(
        opt.resources["expectation"], abs=1e-4
    )
    assert fixed.success_probability() == pytest.approx(opt.success_probability(), abs=1e-9)
    assert fixed.resources["evaluations"] == 1  # no optimizer ran


def test_fixed_angles_validate_depth(islanding):
    with pytest.raises(ValueError, match="p=2"):
        qg.solve(islanding, solver="qaoa", p=2, angles=([0.1], [0.2]))
