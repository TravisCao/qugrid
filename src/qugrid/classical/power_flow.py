"""Classical power flow: DC (linear) and Newton-Raphson AC.

Both follow the MATPOWER formulation, so results are directly comparable to
what researchers get from ``rundcpf`` / ``runpf``. Dense linear algebra keeps
the code readable; the target scale (up to a few hundred buses) does not need
sparsity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qugrid import idx
from qugrid.network import Network


@dataclass
class DCResult:
    """DC power flow solution."""

    theta: np.ndarray  # bus voltage angles [rad]
    flow_mw: np.ndarray  # branch active flows, from-end [MW]
    slack_p_mw: float  # active power picked up by the slack bus [MW]
    loading: np.ndarray  # |flow| / RATE_A where a rating is given, else nan

    def max_loading(self) -> float:
        vals = self.loading[np.isfinite(self.loading)]
        return float(vals.max()) if vals.size else float("nan")


def solve_dc(net: Network) -> DCResult:
    """Solve the DC power flow ``Bbus @ theta = P`` exactly.

    This is also the linear system handed to quantum linear solvers
    (:func:`qugrid.problems.dc_power_flow`), so classical and quantum answers
    are comparable angle by angle.
    """
    bbus, bf, pbusinj, pfinj = net.bdc()
    pbus = np.real(net.sbus()) - net.bus[:, idx.GS] / net.baseMVA
    ref = net.ref
    keep = np.setdiff1d(np.arange(net.n_bus), [ref])
    theta = np.zeros(net.n_bus)
    theta[ref] = np.deg2rad(net.bus[ref, idx.VA])
    rhs = pbus[keep] - pbusinj[keep] - bbus[keep, ref] * theta[ref]
    theta[keep] = np.linalg.solve(bbus[np.ix_(keep, keep)], rhs)
    flow_pu = bf @ theta + pfinj
    flow_mw = flow_pu * net.baseMVA
    rate = net.branch[:, idx.RATE_A]
    loading = np.where(rate > 0, np.abs(flow_mw) / np.where(rate > 0, rate, 1.0), np.nan)
    slack_p = (bbus[ref] @ theta + pbusinj[ref]) * net.baseMVA + net.bus[ref, idx.PD]
    return DCResult(theta=theta, flow_mw=flow_mw, slack_p_mw=float(slack_p), loading=loading)


@dataclass
class ACResult:
    """Newton-Raphson AC power flow solution."""

    vm: np.ndarray  # voltage magnitudes [pu]
    va: np.ndarray  # voltage angles [rad]
    converged: bool
    iterations: int
    mismatch_history: list[float] = field(default_factory=list)
    s_from_mva: np.ndarray | None = None  # complex from-end branch flows [MVA]
    s_to_mva: np.ndarray | None = None

    @property
    def v(self) -> np.ndarray:
        return self.vm * np.exp(1j * self.va)

    def losses_mw(self) -> float:
        return float(np.real(self.s_from_mva + self.s_to_mva).sum())


def _branch_flows(net: Network, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stat = net.branch_on.astype(float)
    ys = stat / (net.branch[:, idx.BR_R] + 1j * net.branch[:, idx.BR_X])
    bc = stat * net.branch[:, idx.BR_B]
    tap_mag = np.where(net.branch[:, idx.TAP] == 0.0, 1.0, net.branch[:, idx.TAP])
    tap = tap_mag * np.exp(1j * np.deg2rad(net.branch[:, idx.SHIFT]))
    ytt = ys + 1j * bc / 2
    yff = ytt / (tap * np.conj(tap))
    yft = -ys / np.conj(tap)
    ytf = -ys / tap
    f, t = net.f_bus, net.t_bus
    if_from = yff * v[f] + yft * v[t]
    it_to = ytf * v[f] + ytt * v[t]
    s_from = v[f] * np.conj(if_from) * net.baseMVA
    s_to = v[t] * np.conj(it_to) * net.baseMVA
    return s_from, s_to


def newton_raphson(net: Network, tol: float = 1e-8, max_iter: int = 20) -> ACResult:
    """Full Newton-Raphson AC power flow in polar coordinates.

    Reference: MATPOWER's ``newtonpf`` (Zimmerman, Murillo-Sanchez, Thomas,
    "MATPOWER: Steady-State Operations, Planning and Analysis Tools for Power
    Systems Research and Education", IEEE Trans. Power Systems, 2011).
    Generator reactive-power limits are not enforced.
    """
    ybus = net.ybus()
    sbus = net.sbus()
    pv, pq = net.pv, net.pq
    pvpq = np.concatenate([pv, pq]).astype(int)

    vm = net.bus[:, idx.VM].copy()
    va = np.deg2rad(net.bus[:, idx.VA])
    on = net.gen_on
    vm[net.gen_bus[on]] = net.gen[on, idx.VG]
    v = vm * np.exp(1j * va)

    history: list[float] = []
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

        dx = np.linalg.solve(jac, -f_vec)
        n1 = len(pvpq)
        va[pvpq] += dx[:n1]
        vm[pq] += dx[n1:]
        v = vm * np.exp(1j * va)

    s_from, s_to = _branch_flows(net, v)
    return ACResult(
        vm=np.abs(v),
        va=np.angle(v),
        converged=converged,
        iterations=it,
        mismatch_history=history,
        s_from_mva=s_from,
        s_to_mva=s_to,
    )
