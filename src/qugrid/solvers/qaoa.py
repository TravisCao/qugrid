"""QAOA — the Quantum Approximate Optimization Algorithm, simulated exactly.

The default quantum solver for QuGrid's combinatorial problems. The
implementation follows Farhi, Goldstone, Gutmann (arXiv:1411.4028): ``p``
alternating layers of cost-phase and transverse-field mixer applied to the
uniform superposition, with angles optimized classically.

What the exact simulation gives research that hardware runs cannot:

* the *full* output distribution, hence exact success probabilities;
* the exact optimality gap at every optimizer step;
* seeds that make every figure reproducible.

Export the same problem via :mod:`qugrid.adapters` when you want shots, noise,
or hardware.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from qugrid.problems.base import CombinatorialProblem
from qugrid.solvers.base import Result, attach_reference, finish_combinatorial, timed
from qugrid.solvers.statevector import (
    apply_rx_all,
    bits_of,
    check_size,
    diag_phase,
    expectation_diag,
    probabilities,
    uniform_state,
)


def _qaoa_state(gammas: np.ndarray, betas: np.ndarray, hnorm: np.ndarray, n: int) -> np.ndarray:
    psi = uniform_state(n)
    for g, b in zip(gammas, betas):
        diag_phase(psi, -g * hnorm)
        apply_rx_all(psi, n, b)
    return psi


def solve_qaoa(
    problem: CombinatorialProblem,
    p: int = 2,
    maxiter: int = 400,
    restarts: int = 3,
    seed: int = 0,
    top_k: int = 16,
    **_ignored,
) -> Result:
    """Solve a QUBO-encoded problem with depth-``p`` QAOA (exact statevector).

    The cost Hamiltonian is rescaled to unit spread before exponentiation so
    that angle ranges are problem-independent; reported energies stay in the
    problem's own units. One restart always uses the linear-ramp initial
    angles that work well in practice (Zhou et al., PRX 2020); the rest are
    random.
    """
    ising = problem.qubo.to_ising()
    n = ising.n
    check_size(n, f"QAOA on {type(problem).__name__}")
    hdiag = ising.all_energies()
    spread = float(hdiag.max() - hdiag.min()) or 1.0
    hnorm = (hdiag - hdiag.min()) / spread

    rng = np.random.default_rng(seed)
    res = Result(solver="qaoa", problem=problem)

    def objective(params: np.ndarray) -> float:
        psi = _qaoa_state(params[:p], params[p:], hnorm, n)
        val = expectation_diag(psi, hdiag)
        res.history.append(val)
        return val

    with timed(res.resources):
        best_val, best_params = np.inf, None
        for r in range(restarts):
            if r == 0:
                ramp = (np.arange(p) + 1) / p
                x0 = np.concatenate([2.0 * ramp, 1.0 * (1 - ramp) + 0.1])
            else:
                x0 = rng.uniform(0, np.pi, size=2 * p)
            out = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})
            if out.fun < best_val:
                best_val, best_params = float(out.fun), np.asarray(out.x)

        psi = _qaoa_state(best_params[:p], best_params[p:], hnorm, n)
        probs = probabilities(psi)
        order = np.argsort(probs)[::-1][: max(top_k, 1)]
        res.top_states = [
            (format(k, f"0{n}b")[::-1], float(probs[k]), float(hdiag[k])) for k in order
        ]
        # best candidate: lowest energy among the most probable states
        k_best = min(order, key=lambda k: hdiag[k])
        x = bits_of(int(k_best), n)

    finish_combinatorial(res, problem, x)
    res.resources.update(
        {
            "n_qubits": n,
            "p": p,
            "evaluations": len(res.history),
            "restarts": restarts,
            "seed": seed,
            "expectation": round(best_val, 6),
        }
    )
    return attach_reference(res, problem)
