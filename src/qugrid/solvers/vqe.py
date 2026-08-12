"""VQE on diagonal Hamiltonians with a hardware-efficient ansatz.

For a QUBO, the ground state of the Ising Hamiltonian is a computational
basis state, so VQE here is a variational *sampler* over bitstrings rather
than a chemistry-style eigensolver. It is included because (a) it is the
other workhorse variational algorithm on today's hardware, (b) its trained
``RealAmplitudes`` parameters transfer directly to Qiskit, and (c) comparing
it against QAOA on the same power system problem is a standard experiment.

Reference: Peruzzo et al., Nat. Commun. 2014 (arXiv:1304.3061).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from qugrid.problems.base import CombinatorialProblem
from qugrid.solvers.base import Result, attach_reference, finish_combinatorial, timed
from qugrid.solvers.statevector import (
    bits_of,
    check_size,
    expectation_diag,
    probabilities,
    real_amplitudes_ansatz,
)


def solve_vqe(
    problem: CombinatorialProblem,
    layers: int = 2,
    maxiter: int = 600,
    restarts: int = 3,
    seed: int = 0,
    top_k: int = 16,
    **_ignored,
) -> Result:
    """Minimize the QUBO energy over a real-amplitudes ansatz (exact statevector)."""
    ising = problem.qubo.to_ising()
    n = ising.n
    check_size(n, f"VQE on {type(problem).__name__}")
    hdiag = ising.all_energies()
    n_params = (layers + 1) * n

    rng = np.random.default_rng(seed)
    res = Result(solver="vqe", problem=problem)

    def objective(theta: np.ndarray) -> float:
        psi = real_amplitudes_ansatz(theta, n)
        val = expectation_diag(psi, hdiag)
        res.history.append(val)
        return val

    with timed(res.resources):
        best_val, best_theta = np.inf, None
        for _ in range(restarts):
            theta0 = rng.uniform(-0.4, 0.4, size=n_params) + np.pi / 2
            out = minimize(objective, theta0, method="COBYLA", options={"maxiter": maxiter})
            if out.fun < best_val:
                best_val, best_theta = float(out.fun), np.asarray(out.x)

        psi = real_amplitudes_ansatz(best_theta, n)
        probs = probabilities(psi)
        order = np.argsort(probs)[::-1][: max(top_k, 1)]
        res.top_states = [
            (format(k, f"0{n}b")[::-1], float(probs[k]), float(hdiag[k])) for k in order
        ]
        k_best = min(order, key=lambda k: hdiag[k])
        x = bits_of(int(k_best), n)

    finish_combinatorial(res, problem, x)
    res.resources.update(
        {
            "n_qubits": n,
            "ansatz": f"RealAmplitudes(layers={layers})",
            "parameters": n_params,
            "evaluations": len(res.history),
            "seed": seed,
            "expectation": round(best_val, 6),
        }
    )
    return attach_reference(res, problem)
