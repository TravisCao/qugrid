"""Economic dispatch as a QUBO: what the binary power encoding costs.

WHAT THIS SOLVES
    Economic dispatch sets the output [MW] of each committed unit so that total
    generation meets demand at least cost [$/h]. This instance has two units and
    100 MW of demand:
        cheap   20 - 80 MW, cost 0.02 P^2 + 12 P $/h
        peaker  10 - 50 MW, cost 0.06 P^2 + 25 P $/h
    The continuous optimum is 80.0 MW and 20.0 MW at 1612.00 $/h.

    A QUBO holds binary variables only, so each unit output becomes
    P_g = pmin_g + delta_g * sum_k 2^k b_{g,k} with power_bits bits per unit and
    the quantum delta_g = (pmax_g - pmin_g) / (2**power_bits - 1). Demand is
    enforced by a squared penalty that dominates cost, so the ground state first
    minimizes the distance to 100 MW and then minimizes cost among the states
    that reach that distance. Only power levels on the grid are reachable, so a
    residual balance error is expected and is a property of the encoding, not of
    the solver.

WHY QUANTUM IS STUDIED HERE
    Every quantum optimizer for dispatch consumes this binary encoding, so the
    encoding error sets the accuracy floor no matter how good the solver is;
    this script separates that floor from solver error, which reports of
    quantum dispatch results often merge.

EXPECTED OUTPUT
    One table prints power_bits from 1 to 4 with the reachable power quantum
    [MW], the dispatch [MW], the balance error [MW], the cost [$/h], the cost
    difference against the continuous optimum at 100 MW, and the solver gap
    against the continuous optimum at the power actually served. One figure with
    two panels is written:
      examples/figures/05_economic_dispatch_discretization_error.png
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
from qugrid.classical.dispatch import GenParams  # noqa: E402

FIGDIR = Path(__file__).parent / "figures"
DEMAND = 100.0
BIT_SWEEP = (1, 2, 3, 4)
TOL_SOLVER_GAP = 1e-9  # the exact solver has no gap at the power it serves


def main() -> int:
    qg.viz.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    gens = [
        GenParams("cheap", 20, 80, 0.02, 12.0),
        GenParams("peaker", 10, 50, 0.06, 25.0),
    ]
    p_continuous, cost_continuous = qg.classical.economic_dispatch(gens, DEMAND)

    print(f"demand             {DEMAND:.1f} MW")
    print(f"continuous optimum {np.round(p_continuous, 4)} MW, "
          f"{cost_continuous:.4f} $/h")

    rows = []
    for power_bits in BIT_SWEEP:
        problem = qg.problems.EconomicDispatchQUBO(gens, demand=DEMAND,
                                                   power_bits=power_bits)
        res = qg.solve(problem, solver="exact")
        power = res.decoded["power"]
        served = float(power.sum())
        # The exact optimum for the power this encoding actually serves. The
        # difference against it is solver error and nothing else.
        _, cost_at_served = qg.classical.economic_dispatch(gens, served)
        rows.append(
            {
                "power_bits": power_bits,
                "n_vars": problem.n,
                "quantum_mw": float(problem.delta.min()),
                "power": power,
                "served": served,
                "balance": float(res.decoded["balance_error_mw"]),
                "cost": float(res.decoded["cost"]),
                "cost_offset": float(res.decoded["cost"]) - cost_continuous,
                "solver_gap": (float(res.decoded["cost"]) - cost_at_served)
                / cost_at_served,
                "feasible": bool(res.feasible),
            }
        )

    print("\nDiscretization sweep, exact ground state of the QUBO at each encoding")
    print(f"  {'bits':>5} {'vars':>5} {'quantum':>9} {'cheap':>8} {'peaker':>8} "
          f"{'served':>8} {'balance':>9} {'cost':>10} {'cost - cont.':>13} "
          f"{'solver gap':>11}")
    print(f"  {'':>5} {'':>5} {'[MW]':>9} {'[MW]':>8} {'[MW]':>8} {'[MW]':>8} "
          f"{'[MW]':>9} {'[$/h]':>10} {'[$/h]':>13} {'':>11}")
    for row in rows:
        print(f"  {row['power_bits']:>5} {row['n_vars']:>5} {row['quantum_mw']:>9.4f} "
              f"{row['power'][0]:>8.4f} {row['power'][1]:>8.4f} {row['served']:>8.4f} "
              f"{row['balance']:>9.4f} {row['cost']:>10.4f} {row['cost_offset']:>13.4f} "
              f"{row['solver_gap']:>11.2e}")

    print(
        "\n  Read the last two columns together. The cost difference against the\n"
        f"  continuous optimum at {DEMAND:.0f} MW is negative at power_bits = 1 because that\n"
        f"  encoding serves only {rows[0]['served']:.1f} MW: it looks cheap because it does not\n"
        "  meet demand. The solver gap is measured against the continuous optimum\n"
        "  at the power each encoding actually serves, and it is zero at every\n"
        "  power_bits. So the exact solver returns the true optimum of every\n"
        "  encoding, and the whole error in this study belongs to the encoding.\n"
        "  A quantum solver applied here can only add to that floor, never reduce it."
    )

    print(
        "\n  Balance error against the smallest reachable power step. The unit with\n"
        "  the smallest quantum can reach every level of an arithmetic sequence of\n"
        "  that step across its own range, so any demand inside that range sits\n"
        "  within half a quantum of a reachable total. That is the bound below."
    )
    print(f"  {'bits':>5} {'balance [MW]':>13} {'half quantum [MW]':>18} {'ratio':>7}")
    for row in rows:
        half = 0.5 * row["quantum_mw"]
        print(f"  {row['power_bits']:>5} {row['balance']:>13.4f} {half:>18.4f} "
              f"{row['balance'] / half:>7.3f}")

    # -- figure: two panels, one per error source ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    bits = [row["power_bits"] for row in rows]

    axes[0].plot(bits, [row["balance"] for row in rows], "o-",
                 color=qg.viz.PALETTE[0], label="balance error")
    axes[0].plot(bits, [0.5 * row["quantum_mw"] for row in rows], "s--",
                 color=qg.viz.MUTED, label="half power quantum")
    axes[0].set_yscale("log")
    axes[0].set_xticks(bits)
    axes[0].set_xlabel("power_bits per unit")
    axes[0].set_ylabel("|generation - demand| [MW]")
    axes[0].set_title("encoding error: unmet demand")
    axes[0].legend(fontsize=9)

    offsets = [row["cost_offset"] for row in rows]
    colors = [qg.viz.PALETTE[7] if v < 0 else qg.viz.PALETTE[0] for v in offsets]
    axes[1].bar(bits, offsets, width=0.55, color=colors,
                edgecolor=qg.viz.SURFACE, linewidth=1.2)
    axes[1].axhline(0.0, color=qg.viz.INK, lw=1.2)
    axes[1].plot(bits, [row["solver_gap"] * row["cost"] for row in rows], "o-",
                 color=qg.viz.PALETTE[2], label="solver error")
    axes[1].set_xticks(bits)
    axes[1].set_xlabel("power_bits per unit")
    axes[1].set_ylabel(f"cost - continuous optimum at {DEMAND:.0f} MW [$/h]")
    axes[1].set_title("cost offset (red = demand not met)")
    axes[1].legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    path_error = FIGDIR / "05_economic_dispatch_discretization_error.png"
    fig.savefig(path_error, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nfigure             {path_error.name}")
    print(f"                   in {FIGDIR}")

    # -- self-validation -----------------------------------------------------
    balances = [row["balance"] for row in rows]
    print("\nvalidation")
    print(f"  balance error [MW]           {[round(v, 4) for v in balances]}")
    print(f"  largest solver gap           "
          f"{max(abs(row['solver_gap']) for row in rows):.3e}")

    for row in rows:
        assert row["balance"] <= 0.5 * row["quantum_mw"] + 1e-9, (
            f"power_bits={row['power_bits']}: balance error {row['balance']:.4f} MW "
            f"exceeds half the power quantum {0.5 * row['quantum_mw']:.4f} MW"
        )
        assert abs(row["solver_gap"]) < TOL_SOLVER_GAP, (
            f"power_bits={row['power_bits']}: the exact solver is not optimal at the "
            f"power it serves, gap {row['solver_gap']:.3e}"
        )
        assert row["feasible"], f"power_bits={row['power_bits']}: infeasible ground state"
    assert all(a > b for a, b in zip(balances, balances[1:])), (
        f"balance error is not strictly decreasing in power_bits: {balances}"
    )
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
