"""AC power flow with a variational quantum linear solver inside Newton-Raphson.

WHAT
    Newton-Raphson AC power flow solves one linear system J dx = -F per
    iteration. This script replaces that inner solve with the Variational
    Quantum Linear Solver of Bravo-Prieto et al. (arXiv:1909.05820) on the
    3-bus microgrid toy3, and compares the result against the exact
    Newton-Raphson reference: voltage magnitudes, voltage angles, iteration
    count, and the outer mismatch trajectory.

WHY QUANTUM
    Power flow is the standard entry point for quantum linear solvers, because
    the Jacobian system is the only expensive step of the iteration. The open
    question is whether an approximate inner solve still gives the outer
    Newton iteration its quadratic convergence. This script measures that. It
    claims no speed advantage: the Jacobian here is 3 x 3, and the variational
    solve costs about a second against microseconds for a dense LU
    factorization.

EXPECTED OUTPUT
    Both methods converge in 4 iterations to the same operating point, with a
    largest voltage magnitude difference of 2.0e-10 pu and a largest angle
    difference of 8.4e-8 degrees. The inner VQLS solutions carry a relative
    error of 1.5e-3 to 6.6e-5, which is 7 orders of magnitude looser than the
    final answer: the outer Newton iteration absorbs the inner error, which is
    the defining property of an inexact Newton method. Two figures are written
    to examples/figures/. Runtime is about 4 seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import qugrid as qg  # noqa: E402
from qugrid.viz import PALETTE, use_style  # noqa: E402

TOL = 1e-6
SEED = 0
VM_TOLERANCE = 1e-4  # pu, the accuracy claim this script validates
FIGDIR = Path(__file__).parent / "figures"


def main() -> int:
    use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    net = qg.cases.toy3()
    reference = qg.classical.newton_raphson(net, tol=TOL)

    inner_log: list[dict] = []

    def vqls_step(linear_problem):
        """Solve one Newton step with VQLS and record how good the step was."""
        result = qg.solve(linear_problem, solver="vqls", seed=SEED)
        inner_log.append(
            {
                "iteration": linear_problem.context["iteration"],
                "dimension": linear_problem.n,
                "qubits": int(result.resources["n_qubits"]),
                "condition": linear_problem.condition_number(),
                "relative_error": result.decoded["relative_error"],
                "fidelity": result.decoded["fidelity_vs_exact"],
                "final_cost": result.decoded["final_cost"],
                "ansatz": result.resources["ansatz"],
                "parameters": int(result.resources["parameters"]),
                "evaluations": int(result.resources["evaluations"]),
                "runtime_s": float(result.resources["wall_time_s"]),
            }
        )
        return result.x

    hybrid = qg.problems.newton_with_linear_solver(net, vqls_step, tol=TOL)

    vm_error = float(np.abs(hybrid.ac.vm - reference.vm).max())
    va_error_deg = float(np.rad2deg(np.abs(hybrid.ac.va - reference.va).max()))

    print(f"network            {net.name}, {net.n_bus} buses, {net.n_branch} branches, "
          f"{net.load_p.sum():.0f} MW load")
    print(f"inner system       {inner_log[0]['dimension']} unknowns, padded to "
          f"{2 ** inner_log[0]['qubits']} amplitudes on {inner_log[0]['qubits']} qubits")
    print(f"ansatz             {inner_log[0]['ansatz']}, "
          f"{inner_log[0]['parameters']} parameters\n")

    header = (
        f"{'method':<22}  {'converged':>9}  {'iterations':>10}  {'linear solves':>13}  "
        f"{'final mismatch':>15}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'Newton-Raphson (LU)':<22}  {str(reference.converged):>9}  "
        f"{reference.iterations:>10}  {reference.iterations - 1:>13}  "
        f"{reference.mismatch_history[-1]:>15.3e}"
    )
    print(
        f"{'Newton-Raphson (VQLS)':<22}  {str(hybrid.ac.converged):>9}  "
        f"{hybrid.ac.iterations:>10}  {hybrid.linear_solves:>13}  "
        f"{hybrid.ac.mismatch_history[-1]:>15.3e}"
    )

    print("\nbus voltages in engineering units")
    header = (
        f"{'bus':>4}  {'|V| ref [pu]':>13}  {'|V| VQLS [pu]':>14}  "
        f"{'angle ref [deg]':>16}  {'angle VQLS [deg]':>17}"
    )
    print(header)
    print("-" * len(header))
    for i in range(net.n_bus):
        print(
            f"{int(net.bus[i, 0]):>4}  {reference.vm[i]:>13.6f}  {hybrid.ac.vm[i]:>14.6f}  "
            f"{np.rad2deg(reference.va[i]):>16.5f}  "
            f"{np.rad2deg(hybrid.ac.va[i]):>17.5f}"
        )
    print(f"\nlargest |V| difference     {vm_error:.2e} pu")
    print(f"largest angle difference   {va_error_deg:.2e} deg")
    print(f"losses, reference          {reference.losses_mw():.4f} MW")
    print(f"losses, VQLS               {hybrid.ac.losses_mw():.4f} MW")

    print("\ninner VQLS solves")
    header = (
        f"{'Newton it':>9}  {'cond(J)':>8}  {'VQLS cost':>10}  {'relative error':>15}  "
        f"{'fidelity':>9}  {'||J dx + F||':>13}  {'evals':>6}  {'time s':>7}"
    )
    print(header)
    print("-" * len(header))
    for log, residual in zip(inner_log, hybrid.inner_residuals):
        print(
            f"{log['iteration']:>9}  {log['condition']:>8.3f}  {log['final_cost']:>10.2e}  "
            f"{log['relative_error']:>15.3e}  {log['fidelity']:>9.6f}  "
            f"{residual:>13.3e}  {log['evaluations']:>6}  {log['runtime_s']:>7.2f}"
        )
    worst_inner = max(log["relative_error"] for log in inner_log)
    print(f"The loosest inner solve is {worst_inner / vm_error:.0e} times less accurate than")
    print("the final answer. The outer Newton iteration removes that error, because a")
    print("Newton step only has to point close enough to the right direction.")

    # ---------------------------------------------------------------- figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9))
    ax1.semilogy(
        range(1, len(reference.mismatch_history) + 1),
        reference.mismatch_history,
        marker="o",
        color=PALETTE[0],
        label="exact inner solve (LU)",
    )
    ax1.semilogy(
        range(1, len(hybrid.ac.mismatch_history) + 1),
        hybrid.ac.mismatch_history,
        marker="s",
        ls="--",
        color=PALETTE[1],
        label="VQLS inner solve",
    )
    ax1.axhline(TOL, color=PALETTE[7], lw=1.2, ls=(0, (3, 3)))
    ax1.annotate(
        f"tolerance {TOL:g}",
        (len(reference.mismatch_history), TOL),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=PALETTE[7],
    )
    ax1.set_xticks(range(1, len(reference.mismatch_history) + 1))
    ax1.set_xlabel("Newton iteration")
    ax1.set_ylabel("power mismatch [pu]")
    ax1.set_title("Outer convergence is unchanged")
    ax1.legend(fontsize=9)

    iterations = [log["iteration"] for log in inner_log]
    ax2.semilogy(
        iterations,
        [log["relative_error"] for log in inner_log],
        marker="o",
        color=PALETTE[2],
        label="relative error of dx",
    )
    ax2.semilogy(
        iterations,
        hybrid.inner_residuals,
        marker="s",
        ls="--",
        color=PALETTE[3],
        label="||J dx + F||",
    )
    ax2.set_xticks(iterations)
    ax2.set_xlabel("Newton iteration")
    ax2.set_ylabel("inner solve error")
    ax2.set_title("The inner solves stay approximate")
    ax2.legend(fontsize=9)
    fig.suptitle(f"Hybrid Newton-Raphson on {net.name}, VQLS inner solver, seed {SEED}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "10_hybrid_newton_vqls_convergence.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    qg.viz.plot_network(
        net,
        node_values=hybrid.ac.vm,
        node_label="|V| [pu], VQLS inner solver",
        ax=ax,
        title=f"{net.name} operating point from the hybrid solver",
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "10_hybrid_newton_vqls_network.png", dpi=150)
    plt.close(fig)
    print(f"\nfigures -> {FIGDIR}/10_hybrid_newton_vqls_*.png")

    # ------------------------------------------------------------- validation
    assert hybrid.ac.converged, "the hybrid solver did not converge"
    assert vm_error < VM_TOLERANCE, (
        f"voltage magnitudes differ by {vm_error:.2e} pu, above {VM_TOLERANCE:g} pu"
    )
    assert hybrid.ac.iterations <= reference.iterations + 1, (
        f"the hybrid solver needed {hybrid.ac.iterations} iterations against "
        f"{reference.iterations} for the exact inner solve"
    )
    assert worst_inner > 1e-6, (
        "the inner solves were exact, so this run does not test an inexact "
        "Newton iteration"
    )
    print(f"VALIDATION PASSED: the hybrid solver converged in {hybrid.ac.iterations} "
          f"iterations, matching")
    print(f"the reference operating point to {vm_error:.1e} pu with inner solves that are")
    print(f"only {worst_inner:.1e} accurate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
