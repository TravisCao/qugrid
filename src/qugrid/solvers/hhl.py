"""HHL — the Harrow-Hassidim-Lloyd quantum linear system algorithm.

Solves ``A x = b`` on a simulated gate model, faithfully to the textbook
circuit: state preparation, quantum phase estimation (QPE) with ``m`` clock
qubits, the controlled eigenvalue-inversion rotation on an ancilla,
inverse QPE, and postselection on the ancilla. Registers are assembled as
dense matrices — exact, transparent, and honest about every error source:

* **Phase discretization**: eigenvalues that do not sit on the ``m``-bit grid
  leak amplitude; watch ``clock_leakage``.
* **Postselection cost**: ``success_probability`` is the fraction of runs a
  hardware experiment would keep.

Negative eigenvalues (which arise from the Hermitian dilation of
non-symmetric systems) are handled with the standard two's-complement phase
convention. The rotation constant defaults to the smallest representable
eigenvalue magnitude on the clock grid — no oracle knowledge of the true
spectrum is assumed.

References: Harrow, Hassidim, Lloyd, PRL 2009 (arXiv:0811.3171); power
system application: Feng, Zhou, Zhang, "Quantum Power Flow", IEEE TPWRS 2021.
"""

from __future__ import annotations

import numpy as np

from qugrid.problems.base import LinearSystemProblem
from qugrid.solvers.base import Result, timed


def _hadamard(m: int) -> np.ndarray:
    h1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    h = np.array([[1.0]])
    for _ in range(m):
        h = np.kron(h, h1)
    return h


def _qft(m: int) -> np.ndarray:
    dim = 2**m
    k = np.arange(dim)
    return np.exp(2j * np.pi * np.outer(k, k) / dim) / np.sqrt(dim)


def _grid_eigenvalue(k: int, m: int) -> float:
    """Two's-complement reading of clock value ``k`` as a signed phase."""
    phase = k / 2**m
    return phase if phase < 0.5 else phase - 1.0


def solve_hhl(
    problem: LinearSystemProblem,
    n_clock: int = 6,
    c_const: float | None = None,
    **_ignored,
) -> Result:
    """Run the HHL circuit on ``problem`` and return the rescaled solution.

    Non-Hermitian systems are embedded via the Hermitian dilation
    ``[[0, A], [A^T, 0]]`` automatically. The classical solution is computed
    once at the end for fidelity reporting — it never steers the circuit.
    """
    res = Result(solver="hhl", problem=problem)
    with timed(res.resources):
        a_pad, b_pad, n_orig = problem.padded()
        dilated = False
        if not np.allclose(a_pad, a_pad.T):
            dim = a_pad.shape[0]
            a_h = np.zeros((2 * dim, 2 * dim))
            a_h[:dim, dim:] = a_pad
            a_h[dim:, :dim] = a_pad.T
            b_h = np.concatenate([b_pad, np.zeros(dim)])
            a_pad, b_pad = a_h, b_h
            dilated = True

        n_sys = int(np.log2(a_pad.shape[0]))
        m = int(n_clock)
        dim_sys, dim_clk = 2**n_sys, 2**m

        # -- scale A so |lambda| t / (2 pi) lands strictly inside (-1/2, 1/2)
        eigmax = float(np.linalg.norm(a_pad, 2))
        t = 2 * np.pi * (2 ** (m - 1) - 1) / (2 ** (m - 1)) / (2 * eigmax)
        u_step = _matexp_unitary(a_pad, t)

        # -- registers: |clock> (x) |system>, ancilla handled as a 2-block
        b_norm = np.linalg.norm(b_pad)
        psi_sys = b_pad / b_norm
        psi = np.kron(np.full(dim_clk, 1 / np.sqrt(dim_clk)), psi_sys).astype(complex)
        # ^ H^m |0> on the clock, |b> on the system

        # -- controlled powers U^k (QPE core), then inverse QFT on the clock
        u_pow = np.eye(dim_sys, dtype=complex)
        blocks = []
        for _k in range(dim_clk):
            blocks.append(u_pow.copy())
            u_pow = u_pow @ u_step
        psi = psi.reshape(dim_clk, dim_sys)
        for k in range(dim_clk):
            psi[k] = blocks[k] @ psi[k]
        qft_dag = _qft(m).conj().T
        psi = (qft_dag @ psi).reshape(dim_clk, dim_sys)

        # -- ancilla rotation: |1> amplitude C / lambda(k)
        c = c_const if c_const is not None else 1.0 / dim_clk  # smallest grid |lambda|
        anc1 = np.zeros(dim_clk)
        for k in range(dim_clk):
            lam = _grid_eigenvalue(k, m)
            anc1[k] = 0.0 if lam == 0.0 else np.clip(c / lam, -1.0, 1.0)
        anc0 = np.sqrt(1.0 - anc1**2)
        psi0 = psi * anc0[:, None]  # ancilla |0> branch
        psi1 = psi * anc1[:, None]  # ancilla |1> branch

        # -- inverse QPE on both branches
        qft_m = _qft(m)
        for branch in (psi0, psi1):
            branch[:] = qft_m @ branch
            for k in range(dim_clk):
                branch[k] = blocks[k].conj().T @ branch[k]
        h_m = _hadamard(m)
        psi1 = h_m @ psi1

        # -- postselect ancilla = 1, read the clock = |0> block
        success_probability = float(np.sum(np.abs(psi1) ** 2))
        sol_state = psi1[0]
        clock_leakage = 0.0
        if success_probability > 0:
            clock_leakage = 1.0 - float(np.sum(np.abs(sol_state) ** 2)) / success_probability

        # -- classical rescaling back to engineering units
        lam_unit = t / (2 * np.pi)  # grid eigenvalue -> physical eigenvalue factor
        x_scaled = np.real(sol_state) / c * lam_unit * b_norm
        if dilated:
            x_scaled = x_scaled[len(x_scaled) // 2 :]
        x = x_scaled[:n_orig]

    x_exact = problem.solve_exact()
    denom = np.linalg.norm(x) * np.linalg.norm(x_exact)
    fidelity = float(abs(np.dot(x, x_exact)) / denom) if denom > 0 else 0.0
    rel_err = float(np.linalg.norm(x - x_exact) / max(np.linalg.norm(x_exact), 1e-30))

    res.x = x
    res.objective = float(np.linalg.norm(problem.a @ x - problem.b))
    res.feasible = rel_err < 0.05
    res.decoded = {
        "residual_norm": res.objective,
        "fidelity_vs_exact": fidelity,
        "relative_error": rel_err,
        "success_probability": success_probability,
        "clock_leakage": clock_leakage,
    }
    res.resources.update(
        {
            "n_qubits": int(np.log2(a_pad.shape[0])) + n_clock + 1,
            "clock_qubits": n_clock,
            "system_qubits": int(np.log2(a_pad.shape[0])),
            "hermitian_dilation": dilated,
            "condition_number": round(problem.condition_number(), 3),
        }
    )
    res.reference = {"x": x_exact, "objective": 0.0}
    return res


def _matexp_unitary(a: np.ndarray, t: float) -> np.ndarray:
    """``exp(i A t)`` for Hermitian ``A`` via eigendecomposition.

    On hardware this unitary is compiled by Hamiltonian simulation; here it is
    exact, which isolates HHL's own error sources (phase discretization,
    postselection) from Trotter error.
    """
    w, v = np.linalg.eigh(a)
    return (v * np.exp(1j * w * t)) @ v.conj().T
