"""The numerical contracts everything else relies on.

If these fail, no downstream result can be trusted; they are deliberately
strict (1e-9 tolerances on exact algebra).
"""

import numpy as np
import pytest

import qugrid as qg
from qugrid.problems.base import QUBO, LinearSystemProblem
from qugrid.problems.builder import QUBOBuilder


def test_qubo_ising_energies_identical():
    rng = np.random.default_rng(7)
    for n in (2, 5, 9):
        qubo = QUBO(q=rng.normal(size=(n, n)), offset=rng.normal())
        ising = qubo.to_ising()
        # invariant every solver and adapter relies on: J strictly upper triangular
        assert np.array_equal(ising.j, np.triu(ising.j, k=1))
        for _ in range(30):
            x = rng.integers(0, 2, size=n)
            assert qubo.energy(x) == pytest.approx(ising.energy(1 - 2 * x), abs=1e-9)


def test_all_energies_matches_direct_evaluation():
    rng = np.random.default_rng(3)
    qubo = QUBO(q=rng.normal(size=(6, 6)), offset=0.5)
    ising = qubo.to_ising()
    energies = ising.all_energies()
    for k in range(2**6):
        x = qubo.bits_from_index(k)
        assert energies[k] == pytest.approx(qubo.energy(x), abs=1e-9)


def test_builder_squared_penalty_expansion():
    rng = np.random.default_rng(11)
    bld = QUBOBuilder()
    idx = [bld.var(f"v{i}") for i in range(5)]
    coefs = rng.normal(size=5)
    const = rng.normal()
    weight = 2.7
    bld.add_squared_penalty(list(zip(idx, coefs)), const, weight)
    qubo = bld.build()
    for _ in range(40):
        x = rng.integers(0, 2, size=5)
        direct = weight * (coefs @ x + const) ** 2
        assert qubo.energy(x) == pytest.approx(direct, abs=1e-9)


def test_batch_energy_matches_scalar():
    rng = np.random.default_rng(5)
    qubo = QUBO(q=rng.normal(size=(7, 7)), offset=-2.0)
    xs = rng.integers(0, 2, size=(20, 7))
    batch = qubo.energy(xs)
    for row, e in zip(xs, batch):
        assert e == pytest.approx(qubo.energy(row), abs=1e-9)


def test_linear_padding_is_inert():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(3, 3)) + 4 * np.eye(3)
    b = rng.normal(size=3)
    prob = LinearSystemProblem(a=a, b=b)
    a_pad, b_pad, n = prob.padded()
    assert a_pad.shape == (4, 4) and n == 3
    x_pad = np.linalg.solve(a_pad, b_pad)
    assert x_pad[:3] == pytest.approx(np.linalg.solve(a, b), abs=1e-10)
    assert x_pad[3] == pytest.approx(0.0, abs=1e-12)


def test_statevector_conventions():
    """Bit i of the basis index is variable/qubit i, in states and energies."""
    from qugrid.solvers.statevector import bits_of, real_amplitudes_ansatz, zero_state

    psi = zero_state(3)
    assert psi[0] == 1.0
    assert list(bits_of(5, 3)) == [1, 0, 1]  # 5 = 0b101 -> x0=1, x1=0, x2=1
    theta = np.zeros((2, 3))
    theta[0, 1] = np.pi  # flip qubit 1 only
    psi = real_amplitudes_ansatz(theta, 3)
    assert np.abs(psi[0b010]) == pytest.approx(1.0, abs=1e-12)


def test_statevector_norm_preserved():
    from qugrid.solvers.statevector import (
        apply_cz_chain,
        apply_h_all,
        apply_rx_all,
        apply_ry,
        uniform_state,
    )

    psi = uniform_state(5)
    apply_rx_all(psi, 5, 0.7)
    apply_ry(psi, 5, 2, 1.1)
    apply_cz_chain(psi, 5)
    apply_h_all(psi, 5)
    assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-12)


def test_qaoa_state_matches_bruteforce_unitary():
    """One QAOA layer against explicit matrix algebra on 2 qubits."""
    from qugrid.solvers.mixers import XMixer
    from qugrid.solvers.qaoa import _qaoa_state

    h = np.array([0.3, -1.2, 0.7, 2.0])
    gamma, beta = 0.9, 0.4
    psi = _qaoa_state(np.array([gamma]), np.array([beta]), h, 2, XMixer())

    x = np.array([[0, 1], [1, 0]])
    rx = np.cos(beta) * np.eye(2) - 1j * np.sin(beta) * x
    mixer = np.kron(rx, rx)  # qubit 1 (high bit) kron qubit 0 (low bit)
    expected = mixer @ (np.exp(-1j * gamma * h) * np.full(4, 0.5))
    assert psi == pytest.approx(expected, abs=1e-12)


def test_solve_dispatch_and_errors(case9):
    prob = qg.problems.Islanding(case9)
    with pytest.raises(ValueError, match="options"):
        qg.solve(prob, solver="hhl")
    with pytest.raises(TypeError):
        qg.solve(object())
    res = qg.solve(prob, solver="exact")
    assert res.feasible
