"""Seeded end-to-end checks of every quantum solver against ground truth."""

import numpy as np
import pytest

import qugrid as qg
from qugrid.classical.dispatch import GenParams


def small_uc():
    gens = [
        GenParams("a", pmin=20, pmax=80, c2=0.02, c1=12.0, startup=50.0),
        GenParams("b", pmin=10, pmax=50, c2=0.05, c1=20.0, startup=30.0),
    ]
    return qg.problems.UnitCommitment(gens, demand=[75.0], power_bits=2)


def test_sa_finds_exact_optimum_on_uc_and_islanding(case9):
    for prob in (small_uc(), qg.problems.Islanding(case9)):
        res = qg.solve(prob, solver="sa", seed=2)
        assert res.gap() == pytest.approx(0.0, abs=1e-9)
        assert res.feasible


def test_qaoa_reaches_optimum_small_uc():
    res = qg.solve(small_uc(), solver="qaoa", seed=0, p=2)
    assert res.feasible
    assert res.gap() == pytest.approx(0.0, abs=1e-6)
    assert res.resources["n_qubits"] == 6
    sp = res.success_probability()
    assert sp is not None and sp > 0.01


def test_qaoa_expectation_improves_with_depth(case9):
    prob = qg.problems.Islanding(case9)
    exps = []
    for p in (1, 3):
        res = qg.solve(prob, solver="qaoa", seed=0, p=p, restarts=2, maxiter=300)
        exps.append(res.resources["expectation"])
    assert exps[1] <= exps[0] + 1e-6


def test_vqe_reaches_optimum_small_uc():
    res = qg.solve(small_uc(), solver="vqe", seed=1, layers=2)
    assert res.gap() == pytest.approx(0.0, abs=1e-6)


def test_random_baseline_worse_or_equal(case9):
    prob = qg.problems.Islanding(case9)
    rnd = qg.solve(prob, solver="random", seed=0, samples=64)
    exact = qg.solve(prob, solver="exact")
    assert rnd.objective >= exact.objective - 1e-9


def test_hhl_error_decreases_with_clock_bits(toy3):
    """More clock bits => finer eigenvalue grid => smaller error.

    Even clock counts only: toy3's eigenvalue ratio is ~1/3, and for odd m the
    scaled phase (2^(m-1)-1)/3 lands exactly on the QPE grid (4^k - 1 = 0 mod
    3), collapsing the error to an m-independent resonance floor. That physics
    is real (and worth teaching) but it is not the monotone trend under test.
    """
    lin = qg.problems.dc_power_flow(toy3)
    errs = []
    for m in (4, 6, 8):
        res = qg.solve(lin, solver="hhl", n_clock=m)
        errs.append(res.decoded["relative_error"])
        assert res.decoded["fidelity_vs_exact"] > 0.999
    assert errs[2] < errs[1] < errs[0]
    assert errs[2] < 5e-3


def test_hhl_dilation_handles_nonsymmetric():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3, 3)) + 3 * np.eye(3)
    prob = qg.problems.LinearSystemProblem(a=a, b=rng.normal(size=3))
    res = qg.solve(prob, solver="hhl", n_clock=9)
    assert res.resources["hermitian_dilation"] is True
    assert res.decoded["relative_error"] < 2e-2


def test_vqls_solves_dc_power_flow(toy3):
    lin = qg.problems.dc_power_flow(toy3)
    res = qg.solve(lin, solver="vqls", seed=0)
    assert res.decoded["fidelity_vs_exact"] > 0.999
    assert res.decoded["relative_error"] < 1e-2
    theta = qg.problems.angles_from_solution(lin, res.x)
    assert theta == pytest.approx(qg.classical.solve_dc(toy3).theta, abs=2e-3)


def test_quantum_kernel_properties():
    from qugrid.solvers import quantum_kernel, scale_features

    rng = np.random.default_rng(0)
    x = scale_features(rng.normal(size=(12, 3)))
    k = quantum_kernel(x)
    assert np.allclose(np.diag(k), 1.0, atol=1e-9)
    assert np.allclose(k, k.T, atol=1e-9)
    eig = np.linalg.eigvalsh(0.5 * (k + k.T))
    assert eig.min() > -1e-9


def test_kernel_classifier_beats_chance_on_screening(case9):
    from qugrid.solvers import compare_kernels, scale_features

    ds = qg.problems.screening_dataset(case9, n_samples=160, seed=1)
    assert 0.3 < ds.y.mean() < 0.7  # classes balanced enough to be meaningful
    xtr, ytr, xte, yte = ds.split(seed=1)
    out = compare_kernels(scale_features(xtr), ytr, scale_features(xte), yte)
    assert out["quantum"]["test_accuracy"] > 0.8
    assert out["rbf"]["test_accuracy"] > 0.8
    assert out["n_qubits"] == 3


def test_qbm_learns_toy_wind_statistics():
    from qugrid.problems import binarize, empirical_statistics, toy_wind_profiles
    from qugrid.solvers import QuantumBoltzmannMachine

    data = binarize(toy_wind_profiles(n_profiles=300, horizon=5, seed=0))
    qbm = QuantumBoltzmannMachine(n_visible=5, gamma=0.8, seed=0)
    errors = qbm.fit(data, epochs=150, lr=0.15)
    assert errors[-1] < 0.25 * errors[0]  # moment mismatch shrinks
    stats = qbm.compare_statistics(data, n_samples=4000, seed=2)
    assert stats["model"]["means"] == pytest.approx(stats["data"]["means"], abs=0.08)
    d = empirical_statistics(data)["neighbor_corr"]
    m = stats["model"]["neighbor_corr"]
    assert np.all(np.sign(m[np.abs(d) > 0.2]) == np.sign(d[np.abs(d) > 0.2]))


def test_result_summary_and_repr(case9):
    res = qg.solve(qg.problems.Islanding(case9), solver="exact")
    text = res.summary()
    assert "objective" in text and "feasible" in text
    assert "Result(" in repr(res)
