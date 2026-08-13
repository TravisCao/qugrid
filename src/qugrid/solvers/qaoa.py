"""QAOA — the Quantum Approximate Optimization Algorithm, simulated exactly.

The default quantum solver for QuGrid's combinatorial problems. The
implementation follows Farhi, Goldstone, Gutmann (arXiv:1411.4028): ``p``
alternating layers of cost-phase and mixer applied to the mixer's ground
state, with angles optimized classically. The mixer is pluggable
(:mod:`qugrid.solvers.mixers`): the standard transverse field, the
warm-start mixer of Egger, Marecek, Woerner (Quantum 5, 479, 2021), or the
one-hot-preserving XY mixer of Wang, Hadfield, Jiang, Rieffel (Phys. Rev. A
101, 012320, 2020).

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
from qugrid.solvers.mixers import WarmStartMixer, XMixer, XYMixer
from qugrid.solvers.statevector import (
    bits_of,
    check_size,
    diag_phase,
    expectation_diag,
    probabilities,
)


def relaxed_solution(qubo) -> np.ndarray:
    """Heuristic box relaxation: minimize ``x^T Q x`` over ``[0, 1]^n``.

    L-BFGS-B from the center of the box. Penalty-folded QUBOs are usually
    nonconvex, so this is a local solution — a useful warm start, not the
    convex QP/SDP relaxation of Egger et al. Pass your own vector to
    ``warm_start=`` when you have a better relaxation (an LP relaxation of
    the original problem, a rounded classical schedule, ...).
    """
    q = qubo.q
    n = qubo.n
    out = minimize(
        lambda x: float(x @ q @ x),
        np.full(n, 0.5),
        jac=lambda x: 2.0 * (q @ x),
        bounds=[(0.0, 1.0)] * n,
        method="L-BFGS-B",
    )
    return np.asarray(out.x)


def _qaoa_state(
    gammas: np.ndarray, betas: np.ndarray, hnorm: np.ndarray, n: int, mixer
) -> np.ndarray:
    psi = mixer.initial_state(n)
    for g, b in zip(gammas, betas):
        diag_phase(psi, -g * hnorm)
        mixer.apply(psi, n, b)
    return psi


def solve_qaoa(
    problem: CombinatorialProblem,
    p: int = 2,
    maxiter: int = 400,
    restarts: int = 3,
    seed: int = 0,
    top_k: int = 16,
    warm_start: np.ndarray | str | None = None,
    epsilon: float = 0.25,
    mixer: XMixer | WarmStartMixer | XYMixer | None = None,
    angles: tuple[np.ndarray, np.ndarray] | None = None,
    **_ignored,
) -> Result:
    """Solve a QUBO-encoded problem with depth-``p`` QAOA (exact statevector).

    The cost Hamiltonian is rescaled to unit spread before exponentiation so
    that angle ranges are problem-independent; reported energies stay in the
    problem's own units. One restart always uses the linear-ramp initial
    angles that work well in practice (Zhou et al., PRX 2020); the rest are
    random.

    ``warm_start`` switches to warm-start QAOA (Egger, Marecek, Woerner,
    Quantum 5, 479, 2021): pass a vector in ``[0, 1]^n`` — a relaxation, a
    rounded classical solution — or ``"relaxation"`` to use
    :func:`relaxed_solution`. ``epsilon`` clips the vector away from 0 and 1
    so no bit freezes. ``mixer`` accepts any object from
    :mod:`qugrid.solvers.mixers` instead (an :class:`XYMixer` built from the
    problem's one-hot groups keeps one-hot constraints exact); ``"xy"``
    builds it from ``problem.one_hot_groups`` when the formulation declares
    them. ``warm_start`` and ``mixer`` are mutually exclusive.

    ``angles=(gammas, betas)`` skips the classical optimizer and evaluates
    the circuit at fixed angles — the angle-transfer setting of Jing, Wang,
    Li (Communications Engineering 2, 12, 2023), who reuse angles across
    IEEE 24-bus maximum-power-section instances keyed by normalized graph
    density. Every optimized run reports its own best angles in
    ``resources["gammas"]/["betas"]``, so transfer experiments can harvest
    them. Caveat: the published transfer evidence is MaxCut-like; on
    penalty-stretched formulations it is untested — measure before trusting.
    """
    if warm_start is not None and mixer is not None:
        raise ValueError("pass either warm_start or mixer, not both")
    ising = problem.qubo.to_ising()
    n = ising.n
    check_size(n, f"QAOA on {type(problem).__name__}")

    if warm_start is not None:
        c = relaxed_solution(problem.qubo) if isinstance(warm_start, str) else warm_start
        if isinstance(warm_start, str) and warm_start != "relaxation":
            raise ValueError(f"unknown warm start {warm_start!r}; pass a vector or 'relaxation'")
        mixer = WarmStartMixer(np.asarray(c, dtype=float), epsilon=epsilon)
    elif mixer == "xy":
        groups = getattr(problem, "one_hot_groups", None)
        if groups is None:
            raise ValueError(
                f"{type(problem).__name__} declares no one_hot_groups; "
                "pass mixer=XYMixer(groups) explicitly"
            )
        mixer = XYMixer(groups)
    elif mixer is None:
        mixer = XMixer()

    hdiag = ising.all_energies()
    spread = float(hdiag.max() - hdiag.min()) or 1.0
    hnorm = (hdiag - hdiag.min()) / spread

    rng = np.random.default_rng(seed)
    res = Result(solver="qaoa", problem=problem)

    def objective(params: np.ndarray) -> float:
        psi = _qaoa_state(params[:p], params[p:], hnorm, n, mixer)
        val = expectation_diag(psi, hdiag)
        res.history.append(val)
        return val

    with timed(res.resources):
        if angles is not None:
            gammas, betas = (np.atleast_1d(np.asarray(a, dtype=float)) for a in angles)
            if len(gammas) != p or len(betas) != p:
                raise ValueError(f"angles must provide {p} gammas and {p} betas for p={p}")
            best_params = np.concatenate([gammas, betas])
            best_val = objective(best_params)
        else:
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

        psi = _qaoa_state(best_params[:p], best_params[p:], hnorm, n, mixer)
        probs = probabilities(psi)
        order = np.argsort(probs)[::-1][: max(top_k, 1)]
        # zero-probability states are unmeasurable and never candidates (an
        # XY mixer puts exactly zero mass outside the one-hot subspace)
        order = [k for k in order if probs[k] > 1e-12] or [int(np.argmax(probs))]
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
            "restarts": restarts if angles is None else 0,
            "seed": seed,
            "expectation": round(best_val, 6),
            "mixer": mixer.name,
            "gammas": [round(float(g), 6) for g in best_params[:p]],
            "betas": [round(float(b), 6) for b in best_params[p:]],
        }
    )
    return attach_reference(res, problem)
