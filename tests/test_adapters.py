"""Adapters to optional quantum ecosystems: the lazy-import contract, plus
the happy path where the package is installed.

qiskit, dimod, and pennylane are not installed in the base dev environment
(see pyproject.toml's optional-dependencies), so every adapter must fail
loudly with a message naming the pip extra rather than a bare
``ModuleNotFoundError`` -- that is the behavior under test locally. The
happy-path tests are guarded by ``importorskip`` and are skipped here,
exercised instead by the ``extras`` CI job (``uv sync --extra dev --extra
all``).
"""

from __future__ import annotations

import importlib.util

import pytest

import qugrid as qg
from qugrid import adapters


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.fixture()
def problem(toy3):
    return qg.problems.Islanding(toy3)


# ------------------------------------------------------------------- qiskit
@pytest.mark.skipif(_installed("qiskit"), reason="qiskit installed; see happy-path test")
def test_to_qiskit_operator_missing_package_hint(problem):
    with pytest.raises(ImportError) as excinfo:
        adapters.to_qiskit_operator(problem)
    assert "qugrid[qiskit]" in str(excinfo.value)


@pytest.mark.skipif(_installed("qiskit"), reason="qiskit installed; see happy-path test")
def test_to_qiskit_qaoa_missing_package_hint(problem):
    with pytest.raises(ImportError) as excinfo:
        adapters.to_qiskit_qaoa(problem)
    assert "qugrid[qiskit]" in str(excinfo.value)


def test_to_qiskit_operator_happy_path(problem):
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import SparsePauliOp

    op = adapters.to_qiskit_operator(problem)
    assert isinstance(op, SparsePauliOp)
    assert op.num_qubits == problem.n


def test_to_qiskit_qaoa_happy_path(problem):
    pytest.importorskip("qiskit")

    qc, gammas, betas = adapters.to_qiskit_qaoa(problem, p=2)
    assert qc.num_qubits == problem.n
    assert len(gammas) == 2 and len(betas) == 2


def test_bits_from_qiskit_key_matches_little_endian_convention():
    """Pure-numpy helper: no qiskit import needed, so it always runs."""
    from qugrid.adapters.qiskit_adapter import bits_from_qiskit_key

    # qiskit counts keys may be space-separated by register; qubit 0 is
    # rightmost. "1100" reversed (space stripped) -> variable i = bit i.
    bits = bits_from_qiskit_key("11 00")
    assert bits.tolist() == [0, 0, 1, 1]


# -------------------------------------------------------------- dimod/Ocean
@pytest.mark.skipif(_installed("dimod"), reason="dimod installed; see happy-path test")
def test_to_bqm_missing_package_hint(problem):
    with pytest.raises(ImportError) as excinfo:
        adapters.to_bqm(problem)
    assert "qugrid[dwave]" in str(excinfo.value)


@pytest.mark.skipif(_installed("dimod"), reason="dimod installed; see happy-path test")
def test_result_from_sampleset_missing_package_hint(problem):
    with pytest.raises(ImportError) as excinfo:
        adapters.result_from_sampleset(problem, sampleset=None)
    assert "qugrid[dwave]" in str(excinfo.value)


def test_to_bqm_happy_path(problem):
    dimod = pytest.importorskip("dimod")

    bqm = adapters.to_bqm(problem)
    assert isinstance(bqm, dimod.BinaryQuadraticModel)
    assert bqm.num_variables == problem.n
    ref = problem.reference()
    sample = {i: int(b) for i, b in enumerate(ref["x"])}
    assert bqm.energy(sample) == pytest.approx(problem.qubo.energy(ref["x"]), abs=1e-9)


def test_result_from_sampleset_happy_path(problem):
    dimod = pytest.importorskip("dimod")

    bqm = adapters.to_bqm(problem)
    sampleset = dimod.ExactSolver().sample(bqm)  # toy3 -> n=3, exact is cheap
    res = adapters.result_from_sampleset(problem, sampleset)
    assert res.solver.startswith("dimod:")
    assert res.objective == pytest.approx(problem.qubo.energy(res.x), abs=1e-9)
    ref = problem.reference()
    assert res.objective == pytest.approx(ref["objective"], abs=1e-6)
    assert res.resources["num_reads"] == len(sampleset)


# ---------------------------------------------------------------- pennylane
@pytest.mark.skipif(_installed("pennylane"), reason="pennylane installed; see happy-path test")
def test_to_pennylane_hamiltonian_missing_package_hint(problem):
    with pytest.raises(ImportError) as excinfo:
        adapters.to_pennylane_hamiltonian(problem)
    assert "qugrid[pennylane]" in str(excinfo.value)


def test_to_pennylane_hamiltonian_happy_path(problem):
    qml = pytest.importorskip("pennylane")

    h = adapters.to_pennylane_hamiltonian(problem)
    assert isinstance(h, qml.Hamiltonian)
    assert len(h.coeffs) == len(h.ops) > 0
