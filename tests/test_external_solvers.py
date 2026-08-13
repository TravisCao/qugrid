"""External solve-through: vendor solvers behind ordinary registry names.

Mirrors ``test_adapters.py``: the lazy-import contract is what the base dev
environment exercises (no SDKs installed there), and the happy paths are
guarded by ``importorskip``, exercised by the ``extras`` CI job. Every call
goes through :func:`qugrid.solve` so the registry wiring is under test too.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import qugrid as qg


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.fixture()
def problem(toy3):
    return qg.problems.Islanding(toy3)


@pytest.fixture()
def reference(problem):
    return qg.solve(problem, solver="exact")


# ---------------------------------------------------------- lazy-import contract


@pytest.mark.skipif(_installed("dimod"), reason="dimod is installed")
def test_dimod_exact_missing_package_hint(problem):
    with pytest.raises(ImportError, match=r"pip install qugrid\[dwave\]"):
        qg.solve(problem, solver="dimod-exact")


@pytest.mark.skipif(_installed("dwave.samplers"), reason="dwave-samplers is installed")
def test_dwave_sa_missing_package_hint(problem):
    with pytest.raises(ImportError, match=r"pip install qugrid\[dwave\]"):
        qg.solve(problem, solver="dwave-sa")


@pytest.mark.skipif(_installed("qiskit_optimization"), reason="qiskit-optimization is installed")
def test_qiskit_qaoa_missing_package_hint(problem):
    with pytest.raises(ImportError, match=r"pip install qugrid\[qiskit\]"):
        qg.solve(problem, solver="qiskit-qaoa")


# ------------------------------------------------------------------ happy paths


def test_dimod_exact_matches_internal_reference(problem, reference):
    pytest.importorskip("dimod")
    res = qg.solve(problem, solver="dimod-exact")
    assert res.solver == "dimod-exact"
    assert abs(res.objective - reference.objective) < 1e-6
    # toy3 is a complete triangle: even the QUBO optimum is an infeasible
    # islanding. The two enumerations must agree on that verdict too.
    assert res.feasible == reference.feasible
    assert res.gap() == pytest.approx(0.0, abs=1e-9)
    assert res.resources["wall_time_s"] >= 0.0
    assert res.resources["states_enumerated"] == 2**problem.n


def test_dwave_sa_seeded_and_fields_populated(problem, reference):
    pytest.importorskip("dwave.samplers")
    first = qg.solve(problem, solver="dwave-sa", num_reads=25, seed=7)
    again = qg.solve(problem, solver="dwave-sa", num_reads=25, seed=7)
    assert first.objective == again.objective  # same seed, same answer
    # 8 states, 25 reads: the mature annealer cannot miss the optimum here.
    assert abs(first.objective - reference.objective) < 1e-6
    assert first.feasible == reference.feasible
    assert first.resources["num_reads"] == 25
    assert first.resources["seed"] == 7
    assert first.resources["wall_time_s"] >= 0.0


def test_qiskit_qaoa_fields_and_objective_bound(problem, reference):
    pytest.importorskip("qiskit_optimization")
    pytest.importorskip("qiskit_algorithms")
    res = qg.solve(problem, solver="qiskit-qaoa", reps=1, maxiter=20, seed=1)
    assert res.solver == "qiskit-qaoa"
    # A minimizer cannot beat the exact optimum; it may tie or fall short.
    assert res.objective >= reference.objective - 1e-9
    assert np.isfinite(res.gap())
    assert res.resources["p"] == 1
    assert res.resources["n_qubits"] == problem.n
    assert res.resources["wall_time_s"] >= 0.0
