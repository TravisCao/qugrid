"""VQLS — the Variational Quantum Linear Solver.

The near-term counterpart to HHL: prepare a trial state ``|x(theta)>`` with a
shallow ansatz and minimize a cost that vanishes exactly when
``A|x> ∝ |b>``. QuGrid uses the normalized global cost

    C(theta) = 1 - |<b|A|x(theta)>|^2 / <x|A^T A|x(theta)>

from Bravo-Prieto et al. (arXiv:1909.05820). The docstring caveat from that
paper applies here too: global costs train poorly as systems grow; at power
flow demo sizes (2-16 unknowns) they are fine.

The quantum state fixes only the *direction* of the solution; the scale is
recovered classically by least squares on ``||s * A x_hat - b||`` — the same
trick every experimental linear-solver paper uses.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from qugrid.problems.base import LinearSystemProblem
from qugrid.solvers.base import Result, timed
from qugrid.solvers.statevector import real_amplitudes_ansatz


def solve_vqls(
    problem: LinearSystemProblem,
    layers: int = 3,
    maxiter: int = 800,
    restarts: int = 3,
    seed: int = 0,
    **_ignored,
) -> Result:
    """Minimize the VQLS global cost with a real-amplitudes ansatz."""
    res = Result(solver="vqls", problem=problem)
    with timed(res.resources):
        a_pad, b_pad, n_orig = problem.padded()
        dim = a_pad.shape[0]
        n = int(np.log2(dim))
        b_normed = b_pad / np.linalg.norm(b_pad)
        n_params = (layers + 1) * n

        rng = np.random.default_rng(seed)

        def cost(theta: np.ndarray) -> float:
            x_state = np.real(real_amplitudes_ansatz(theta, n))
            ax = a_pad @ x_state
            denom = float(ax @ ax)
            if denom < 1e-300:
                return 1.0
            val = 1.0 - float(np.dot(b_normed, ax)) ** 2 / denom
            res.history.append(val)
            return val

        best_val, best_theta = np.inf, None
        for _ in range(restarts):
            theta0 = rng.uniform(-0.5, 0.5, size=n_params) + np.pi / 2
            out = minimize(cost, theta0, method="COBYLA", options={"maxiter": maxiter})
            if out.fun < best_val:
                best_val, best_theta = float(out.fun), np.asarray(out.x)

        x_hat = np.real(real_amplitudes_ansatz(best_theta, n))
        ax = a_pad @ x_hat
        scale = float(np.dot(ax, b_pad) / np.dot(ax, ax))  # least-squares scale recovery
        x = (scale * x_hat)[:n_orig]

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
        "final_cost": best_val,
    }
    res.resources.update(
        {
            "n_qubits": n,
            "ansatz": f"RealAmplitudes(layers={layers})",
            "parameters": n_params,
            "evaluations": len(res.history),
            "seed": seed,
            "condition_number": round(problem.condition_number(), 3),
        }
    )
    res.reference = {"x": x_exact, "objective": 0.0}
    return res
