"""Do the QUBO encodings optimize the *power system* quantities they claim to?

Each test compares the QUBO ground state (exact enumeration) against an
independent classical formulation of the same engineering problem.
"""

import numpy as np
import pytest

import qugrid as qg
from qugrid.classical.dispatch import GenParams, economic_dispatch, solve_uc_enumerate

GENS2 = [
    GenParams("cheap", pmin=20, pmax=80, c2=0.02, c1=12.0, c0=60.0, startup=120.0),
    GenParams("peaker", pmin=10, pmax=50, c2=0.06, c1=25.0, c0=40.0, startup=60.0),
]


def test_economic_dispatch_kkt():
    demand = 100.0
    p, cost = economic_dispatch(GENS2, demand)
    assert p.sum() == pytest.approx(demand, abs=1e-6)
    interior = [(g, pi) for g, pi in zip(GENS2, p) if g.pmin + 1e-6 < pi < g.pmax - 1e-6]
    if len(interior) >= 2:  # equal marginal cost at every interior unit
        lams = [g.c1 + 2 * g.c2 * pi for g, pi in interior]
        assert max(lams) - min(lams) < 1e-5
    brute = min(
        sum(g.cost(pv) for g, pv in zip(GENS2, [p1, demand - p1]))
        for p1 in np.linspace(GENS2[0].pmin, GENS2[0].pmax, 4001)
        if GENS2[1].pmin <= demand - p1 <= GENS2[1].pmax
    )
    assert cost == pytest.approx(brute, rel=1e-5)


def test_ed_qubo_ground_state_is_lexicographic():
    """Dominant balance penalty => minimal representable imbalance first,
    then minimal cost among those states (the documented semantics)."""
    prob = qg.problems.EconomicDispatchQUBO(GENS2, demand=100.0, power_bits=3)
    ref = prob.reference()  # exact enumeration of the QUBO
    assert prob.is_feasible(ref["x"])
    n = prob.qubo.n
    all_x = [prob.qubo.bits_from_index(k) for k in range(2**n)]
    residuals = np.array([abs(prob.power(x).sum() - prob.demand) for x in all_x])
    min_r = residuals.min()
    ref_r = abs(prob.power(ref["x"]).sum() - prob.demand)
    assert ref_r == pytest.approx(min_r, abs=1e-9)
    best_at_min = min(
        prob.decode(x)["cost"] for x, r in zip(all_x, residuals) if r < min_r + 1e-9
    )
    assert ref["cost"] == pytest.approx(best_at_min, rel=1e-9)
    # discretization gap vs the continuous optimum stays small
    cont = prob.continuous_reference()
    assert (ref["cost"] - cont["cost"]) / cont["cost"] < 0.10


def test_uc_qubo_matches_enumerated_uc():
    """On a grid-representable instance the QUBO reproduces the true UC optimum
    exactly (commitment, dispatch, and cost)."""
    gens = [
        GenParams("base", pmin=0, pmax=90, c2=0.02, c1=10.0, c0=50.0, startup=100.0),
        GenParams("peaker", pmin=0, pmax=60, c2=0.04, c1=20.0, c0=30.0, startup=80.0),
    ]
    demand = [60.0, 130.0]  # both periods representable: 60 = 2*30, 130 = 90 + 2*20
    uc = qg.problems.UnitCommitment(gens, demand, power_bits=2)
    ref = uc.reference()
    assert uc.is_feasible(ref["x"])
    decoded = uc.decode(ref["x"])
    # commitment pattern must match the true (continuous-dispatch) optimum
    truth = solve_uc_enumerate(gens, demand)
    assert np.array_equal(decoded["commit"], truth.commit)
    # the continuous optimum lies on the discretization grid here => exact match
    assert decoded["cost"] == pytest.approx(truth.cost, rel=1e-6)
    assert decoded["balance_error_mw"] == pytest.approx(0.0, abs=1e-9)
    # penalty terms vanish at the optimum: QUBO energy equals engineering cost
    assert ref["objective"] == pytest.approx(decoded["cost"], rel=1e-6)


def test_uc_startup_cost_matters():
    """With a huge startup cost the optimum avoids cycling the peaker."""
    gens = [
        GenParams("base", pmin=20, pmax=100, c2=0.02, c1=12.0, startup=0.0),
        GenParams("peaker", pmin=5, pmax=60, c2=0.03, c1=13.0, startup=1e4),
    ]
    truth = solve_uc_enumerate(gens, [90.0, 130.0, 90.0])
    assert truth.commit[1].sum() <= 1 or truth.commit[1].tolist() == [1, 1, 1]


def test_islanding_optimum_is_balanced_and_nontrivial(case9):
    prob = qg.problems.Islanding(case9)
    ref = prob.reference()
    d = prob.decode(ref["x"])
    assert prob.is_feasible(ref["x"])  # both islands non-empty
    assert d["islands_connected"] == (True, True)
    total = case9.gen_p_per_bus().sum() - case9.load_p.sum()
    half_imbalance = max(abs(np.array(d["island_power_mw"]) - total / 2))
    assert half_imbalance < 60.0  # islands within 60 MW of perfect split
    assert 1 <= d["n_cut"] <= 4


def test_pmu_qubo_reference_matches_enumeration(case9):
    prob = qg.problems.PMUPlacement(case9)
    ref = prob.reference()
    assert ref["n_uncovered"] == 0
    # WSCC 9-bus: known minimum dominating set size is 3
    assert ref["n_pmu"] == 3
    # QUBO ground state achieves the same PMU count and full coverage
    qubo_best = qg.solve(prob, solver="exact")
    assert qubo_best.feasible
    assert qubo_best.decoded["n_pmu"] == ref["n_pmu"]


def test_dc_power_flow_problem_equals_reference(toy3, case9):
    for net in (toy3, case9):
        lin = qg.problems.dc_power_flow(net)
        x = lin.solve_exact()
        theta = qg.problems.angles_from_solution(lin, x)
        dc = qg.classical.solve_dc(net)
        assert theta == pytest.approx(dc.theta, abs=1e-10)
        flows = qg.problems.flows_from_angles(lin, theta)
        assert flows == pytest.approx(dc.flow_mw, abs=1e-8)


def test_newton_raphson_converges_on_bundled_cases():
    for name in ("case5", "case9", "case14", "case30", "case57", "case118"):
        net = qg.cases.load_case(name)
        ac = qg.classical.newton_raphson(net)
        assert ac.converged, name
        assert ac.mismatch_history[-1] < 1e-8
        assert ac.iterations <= 10
        # physical sanity: voltage magnitudes in a credible band
        assert ac.vm.min() > 0.85 and ac.vm.max() < 1.15, name


def test_hybrid_newton_with_exact_inner_matches_plain(case9):
    plain = qg.classical.newton_raphson(case9, tol=1e-8)
    hybrid = qg.problems.newton_with_linear_solver(
        case9, lambda p: p.solve_exact(), tol=1e-8
    )
    assert hybrid.ac.converged
    assert hybrid.ac.vm == pytest.approx(plain.vm, abs=1e-9)
    assert hybrid.ac.va == pytest.approx(plain.va, abs=1e-9)


def test_screening_dataset_two_classes(case9):
    ds = qg.problems.screening_dataset(case9, n_samples=120, seed=0)
    assert set(np.unique(ds.y)) == {0, 1}
    assert 0.15 < ds.y.mean() < 0.85  # neither class degenerate
    xtr, ytr, xte, yte = ds.split()
    assert len(ytr) + len(yte) == 120
