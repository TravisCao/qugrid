"""DC power flow of a 3-bus microgrid, solved with the HHL quantum linear solver.

WHAT THIS SOLVES
    DC power flow is the linearized steady-state model of a transmission
    network. It solves ``B' theta = P``, where ``B'`` is the reduced bus
    susceptance matrix [pu], ``P`` is the bus active-power injection [pu], and
    ``theta`` is the bus voltage angle [rad]. The toy3 microgrid has 3 buses,
    3 lines, 2 generators (90 MW at bus 1, 60 MW at bus 2), and a 150 MW load
    at bus 3. The slack bus angle is fixed, so the system that remains is
    2 x 2, symmetric, and positive definite. Branch flows [MW] follow from the
    angles.

WHY QUANTUM IS STUDIED HERE
    Every Newton-Raphson iteration of AC power flow also solves one linear
    system, so a quantum linear system algorithm applies to the inner loop of
    any power flow study; at this size the classical solve is immediate, and
    this script measures HHL accuracy against it, not speed.

EXPECTED OUTPUT
    Three tables print:
      1. Bus angles [deg] from numpy, from HHL, and from the classical DC
         reference, with the HHL angle error [deg].
      2. Branch flows [MW] from HHL against the classical DC reference.
      3. HHL relative error, postselection success probability, and qubit
         count against the clock register size.
    Two figures are written:
      examples/figures/01_dc_power_flow_hhl_error.png
      examples/figures/01_dc_power_flow_hhl_network.png
    The script exits with status 0 when every assertion at the end holds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import qugrid as qg  # noqa: E402

FIGDIR = Path(__file__).parent / "figures"
CLOCK_MAIN = 8
CLOCK_SWEEP = (4, 6, 8, 10)
CLOCK_SWEEP_ODD = (5, 7, 9)

# Tolerances. The values are measured, not guessed; see the printout.
TOL_RELATIVE_ERROR = 5e-3  # measured 2.53e-3 at n_clock = 8
TOL_ANGLE_DEG = 0.02  # measured 0.0067 deg at n_clock = 8
TOL_FLOW_MW = 0.5  # measured 0.169 MW at n_clock = 8


def sweep_clock(problem, clock_sizes) -> list[dict]:
    """Run HHL at each clock register size and collect the error metrics."""
    rows = []
    for n_clock in clock_sizes:
        res = qg.solve(problem, solver="hhl", n_clock=n_clock)
        rows.append(
            {
                "n_clock": n_clock,
                "relative_error": res.decoded["relative_error"],
                "fidelity": res.decoded["fidelity_vs_exact"],
                "p_success": res.decoded["success_probability"],
                "clock_leakage": res.decoded["clock_leakage"],
                "n_qubits": res.resources["n_qubits"],
            }
        )
    return rows


def main() -> int:
    qg.viz.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    net = qg.cases.toy3()
    lin = qg.problems.dc_power_flow(net)
    reference = qg.classical.solve_dc(net)

    print(f"network            {net}")
    print(f"linear system      {lin.n} x {lin.n}, condition number "
          f"{lin.condition_number():.4f}")
    print(f"generation [MW]    {np.round(net.gen_p_per_bus(), 1)}")
    print(f"load [MW]          {np.round(net.load_p, 1)}")

    # -- solve the same system classically and with HHL -------------------
    res_numpy = qg.solve(lin, solver="numpy")
    res_hhl = qg.solve(lin, solver="hhl", n_clock=CLOCK_MAIN)

    theta_numpy = qg.problems.angles_from_solution(lin, res_numpy.x)
    theta_hhl = qg.problems.angles_from_solution(lin, res_hhl.x)
    flow_hhl = qg.problems.flows_from_angles(lin, theta_hhl)

    angle_err_deg = np.rad2deg(theta_hhl - reference.theta)
    flow_err_mw = flow_hhl - reference.flow_mw

    print(f"\nBus voltage angles [deg], HHL with n_clock = {CLOCK_MAIN}")
    print(f"  {'bus':>4} {'numpy':>10} {'HHL':>10} {'DC reference':>14} {'error':>10}")
    for i in range(net.n_bus):
        print(f"  {int(net.bus[i, 0]):>4} {np.rad2deg(theta_numpy[i]):>10.5f} "
              f"{np.rad2deg(theta_hhl[i]):>10.5f} "
              f"{np.rad2deg(reference.theta[i]):>14.5f} {angle_err_deg[i]:>10.5f}")

    print("\nBranch active-power flows [MW]")
    print(f"  {'branch':>8} {'HHL':>10} {'DC reference':>14} {'error':>10} {'loading':>9}")
    for line in range(net.n_branch):
        f_bus = int(net.bus[net.f_bus[line], 0])
        t_bus = int(net.bus[net.t_bus[line], 0])
        print(f"  {f_bus:>3} -{t_bus:>3} {flow_hhl[line]:>10.3f} "
              f"{reference.flow_mw[line]:>14.3f} {flow_err_mw[line]:>10.3f} "
              f"{reference.loading[line]:>9.3f}")

    # -- accuracy against the clock register size -------------------------
    even = sweep_clock(lin, CLOCK_SWEEP)
    odd = sweep_clock(lin, CLOCK_SWEEP_ODD)

    print("\nHHL accuracy against the clock register size")
    print(f"  {'n_clock':>8} {'qubits':>7} {'relative error':>15} {'fidelity':>11} "
          f"{'P(success)':>12} {'clock leakage':>14}")
    for row in sorted(even + odd, key=lambda r: r["n_clock"]):
        print(f"  {row['n_clock']:>8} {row['n_qubits']:>7} "
              f"{row['relative_error']:>15.3e} {row['fidelity']:>11.8f} "
              f"{row['p_success']:>12.3e} {row['clock_leakage']:>14.3e}")

    print(
        "\n  Even and odd clock sizes behave differently on this system. The two\n"
        "  eigenvalues of B' have the ratio 3.0006. The phase estimation grid\n"
        "  holds 2**(n_clock-1) - 1 steps below the largest eigenvalue. That\n"
        "  count is divisible by 3 only when n_clock is odd, so at odd sizes both\n"
        "  eigenvalues sit on the grid and the error saturates at the level set by\n"
        "  the ratio being 3.0006 and not exactly 3. The even sizes show the\n"
        "  general behaviour: the error falls as the clock register grows, and the\n"
        "  postselection success probability falls with it."
    )

    # -- figure 1: error and postselection cost against clock size --------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    nc_even = [r["n_clock"] for r in even]
    nc_odd = [r["n_clock"] for r in odd]
    axes[0].plot(nc_even, [r["relative_error"] for r in even], "o-",
                 color=qg.viz.PALETTE[0], label="even n_clock")
    axes[0].plot(nc_odd, [r["relative_error"] for r in odd], "s--",
                 color=qg.viz.PALETTE[1], label="odd n_clock")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("clock qubits n_clock")
    axes[0].set_ylabel("relative error of theta")
    axes[0].set_title("HHL solution error")
    axes[0].legend(fontsize=9)

    axes[1].plot(nc_even, [r["p_success"] for r in even], "o-",
                 color=qg.viz.PALETTE[0], label="even n_clock")
    axes[1].plot(nc_odd, [r["p_success"] for r in odd], "s--",
                 color=qg.viz.PALETTE[1], label="odd n_clock")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("clock qubits n_clock")
    axes[1].set_ylabel("P(ancilla = 1)")
    axes[1].set_title("postselection success probability")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    path_error = FIGDIR / "01_dc_power_flow_hhl_error.png"
    fig.savefig(path_error, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- figure 2: the network with the HHL bus angles --------------------
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    qg.viz.plot_network(
        net,
        node_values=np.rad2deg(theta_hhl),
        node_label="bus angle [deg]",
        edge_loading=reference.loading,
        ax=ax,
        title=f"toy3 DC power flow, HHL with n_clock = {CLOCK_MAIN}",
    )
    path_network = FIGDIR / "01_dc_power_flow_hhl_network.png"
    fig.savefig(path_network, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nfigures            {path_error.name}, {path_network.name}")
    print(f"                   in {FIGDIR}")

    # -- self-validation --------------------------------------------------
    rel_err = res_hhl.decoded["relative_error"]
    max_angle_err = float(np.abs(angle_err_deg).max())
    max_flow_err = float(np.abs(flow_err_mw).max())
    errors_even = [r["relative_error"] for r in even]

    print("\nvalidation")
    print(f"  relative error at n_clock={CLOCK_MAIN}   {rel_err:.3e} "
          f"(limit {TOL_RELATIVE_ERROR:.0e})")
    print(f"  max bus angle error [deg]     {max_angle_err:.3e} "
          f"(limit {TOL_ANGLE_DEG:.0e})")
    print(f"  max branch flow error [MW]    {max_flow_err:.3e} "
          f"(limit {TOL_FLOW_MW:.1e})")

    assert rel_err < TOL_RELATIVE_ERROR, f"HHL relative error {rel_err:.3e}"
    assert max_angle_err < TOL_ANGLE_DEG, f"angle error {max_angle_err:.3e} deg"
    assert max_flow_err < TOL_FLOW_MW, f"flow error {max_flow_err:.3e} MW"
    assert np.allclose(res_numpy.x, lin.solve_exact()), "numpy solver disagrees with numpy"
    assert all(a > b for a, b in zip(errors_even, errors_even[1:])), (
        f"error is not monotone over even clock sizes: {errors_even}"
    )
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
