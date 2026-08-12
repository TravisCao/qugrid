"""Power flow as linear algebra — the doorway to quantum linear solvers.

DC power flow *is* a linear system, and each Newton-Raphson iteration of AC
power flow solves one. That makes power flow the natural power system
application for quantum linear system algorithms (HHL, VQLS), following
Eskandarpour et al., "Quantum-Enhanced Grid of the Future: A Primer", IEEE
Access 2020, and Feng, Zhou, Zhang, "Quantum Power Flow", IEEE Trans. Power
Systems 2021 (arXiv:2104.04888).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qugrid import idx
from qugrid.classical.power_flow import ACResult, _branch_flows
from qugrid.network import Network
from qugrid.problems.base import LinearSystemProblem


def dc_power_flow(net: Network) -> LinearSystemProblem:
    """The reduced DC power flow system ``B' theta = P`` as a solver-ready problem.

    The slack bus row/column are removed (its angle is fixed), leaving a
    symmetric positive-definite system — exactly what HHL-style algorithms
    want. ``context`` carries everything needed to map ``x`` back to angles.
    """
    bbus, bf, pbusinj, pfinj = net.bdc()
    pbus = np.real(net.sbus()) - net.bus[:, idx.GS] / net.baseMVA
    ref = net.ref
    keep = np.setdiff1d(np.arange(net.n_bus), [ref])
    theta_ref = float(np.deg2rad(net.bus[ref, idx.VA]))
    a = bbus[np.ix_(keep, keep)]
    b = pbus[keep] - pbusinj[keep] - bbus[keep, ref] * theta_ref
    names = [f"theta_bus{int(net.bus[i, 0])}" for i in keep]
    return LinearSystemProblem(
        a=a,
        b=b,
        names=names,
        unit="rad",
        context={"net": net, "ref": ref, "keep": keep, "theta_ref": theta_ref, "bf": bf,
                 "pfinj": pfinj},
    )


def angles_from_solution(problem: LinearSystemProblem, x: np.ndarray) -> np.ndarray:
    """Expand a reduced solution back to all-bus angles [rad]."""
    ctx = problem.context
    net: Network = ctx["net"]
    theta = np.zeros(net.n_bus)
    theta[ctx["ref"]] = ctx["theta_ref"]
    theta[ctx["keep"]] = x
    return theta


def flows_from_angles(problem: LinearSystemProblem, theta: np.ndarray) -> np.ndarray:
    """Branch active-power flows [MW] from bus angles."""
    ctx = problem.context
    net: Network = ctx["net"]
    return (ctx["bf"] @ theta + ctx["pfinj"]) * net.baseMVA


@dataclass
class HybridNewtonResult:
    """AC power flow solved with a pluggable (quantum) inner linear solver."""

    ac: ACResult
    linear_solves: int
    inner_residuals: list[float]  # ||J dx + F|| after each inner solve


def newton_with_linear_solver(
    net: Network,
    solve_fn,
    tol: float = 1e-6,
    max_iter: int = 20,
) -> HybridNewtonResult:
    """Newton-Raphson AC power flow with a caller-supplied linear solver.

    ``solve_fn(problem: LinearSystemProblem) -> np.ndarray`` receives each
    Newton step ``J dx = -F`` and returns ``dx``. Pass an exact solver, HHL,
    or VQLS — this is the standard hybrid quantum-classical decomposition of
    AC power flow.
    """
    ybus = net.ybus()
    sbus = net.sbus()
    ref, pv, pq = net.ref, net.pv, net.pq
    del ref
    pvpq = np.concatenate([pv, pq]).astype(int)

    vm = net.bus[:, idx.VM].copy()
    va = np.deg2rad(net.bus[:, idx.VA])
    on = net.gen_on
    vm[net.gen_bus[on]] = net.gen[on, idx.VG]
    v = vm * np.exp(1j * va)

    history: list[float] = []
    inner_residuals: list[float] = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        ibus = ybus @ v
        mis = v * np.conj(ibus) - sbus
        f_vec = np.concatenate([np.real(mis[pvpq]), np.imag(mis[pq])])
        norm = float(np.linalg.norm(f_vec, np.inf)) if f_vec.size else 0.0
        history.append(norm)
        if norm < tol:
            converged = True
            break

        diag_v = np.diag(v)
        diag_i = np.diag(ibus)
        diag_vn = np.diag(v / np.abs(v))
        ds_dva = 1j * diag_v @ np.conj(diag_i - ybus @ diag_v)
        ds_dvm = diag_v @ np.conj(ybus @ diag_vn) + np.conj(diag_i) @ diag_vn
        j11 = np.real(ds_dva[np.ix_(pvpq, pvpq)])
        j12 = np.real(ds_dvm[np.ix_(pvpq, pq)])
        j21 = np.imag(ds_dva[np.ix_(pq, pvpq)])
        j22 = np.imag(ds_dvm[np.ix_(pq, pq)])
        jac = np.block([[j11, j12], [j21, j22]])

        step = LinearSystemProblem(
            a=jac, b=-f_vec, unit="mixed rad/pu", context={"iteration": it}
        )
        dx = np.asarray(solve_fn(step), dtype=float)
        inner_residuals.append(float(np.linalg.norm(jac @ dx + f_vec)))

        n1 = len(pvpq)
        va[pvpq] += dx[:n1]
        vm[pq] += dx[n1:]
        v = vm * np.exp(1j * va)

    s_from, s_to = _branch_flows(net, v)
    ac = ACResult(
        vm=np.abs(v),
        va=np.angle(v),
        converged=converged,
        iterations=it,
        mismatch_history=history,
        s_from_mva=s_from,
        s_to_mva=s_to,
    )
    return HybridNewtonResult(ac=ac, linear_solves=len(inner_residuals),
                              inner_residuals=inner_residuals)
