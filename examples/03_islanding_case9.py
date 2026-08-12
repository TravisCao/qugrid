"""Controlled islanding of the WSCC 9-bus system, solved as a QUBO.

WHAT THIS SOLVES
    Controlled islanding splits a network into self-sufficient parts before a
    disturbance propagates. Each bus gets one binary variable that assigns it to
    island A or island B, so the WSCC 9-bus system needs 9 variables. The
    objective adds three terms:
      cut weight      the summed electrical coupling 1/x [pu] of the opened
                      lines, which the split should keep small;
      power imbalance the squared generation-minus-load mismatch [pu] inside
                      each island, which decides how much generation or load
                      each island must shed after the split;
      size balance    a small term that rules out the answer with no split.
    The 9-bus system carries 315 MW of load and 320.3 MW of scheduled
    generation over 9 branches.

WHY QUANTUM IS STUDIED HERE
    Graph partitioning is the power system problem with the longest record on
    quantum annealers, and its bus assignment maps to one qubit per bus without
    any encoding overhead; at 9 buses exhaustive search is immediate, and this
    script measures the quantum solvers against it.

EXPECTED OUTPUT
    Four tables print: the network summary, the solver comparison, the island
    contents in MW, and the DC power flow across each opened line before the
    split. Two figures are written:
      examples/figures/03_islanding_case9_network.png
      examples/figures/03_islanding_case9_distribution.png
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
SEED = 0
TOL_GAP = 1e-9


def branch_of(net, f: int, t: int) -> int:
    """Index of the branch that joins two internal bus indices."""
    for line in range(net.n_branch):
        pair = {int(net.f_bus[line]), int(net.t_bus[line])}
        if pair == {f, t}:
            return line
    raise ValueError(f"no branch between internal buses {f} and {t}")


def main() -> int:
    qg.viz.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    net = qg.cases.case9()
    problem = qg.problems.Islanding(net)
    dc = qg.classical.solve_dc(net)
    injection_mw = net.gen_p_per_bus() - net.load_p

    print(f"network            {net}")
    print(f"problem            {problem!r}")
    print(f"generation [MW]    {net.gen_p_per_bus().sum():.1f}")
    print(f"load [MW]          {net.load_p.sum():.1f}")
    print(f"max branch loading {dc.max_loading():.3f} of rating")

    # -- solvers ------------------------------------------------------------
    runs = [
        ("exact enumeration", qg.solve(problem, solver="exact", seed=SEED)),
        ("simulated annealing",
         qg.solve(problem, solver="sa", seed=SEED, n_sweeps=1500, n_restarts=8)),
        ("QAOA p=2",
         qg.solve(problem, solver="qaoa", p=2, restarts=3, maxiter=400, seed=SEED)),
    ]

    print("\nSolver comparison")
    print(f"  {'solver':>21} {'objective':>11} {'gap':>10} {'cut lines':>10} "
          f"{'cut weight':>11} {'P(optimum)':>11} {'time [s]':>9}")
    for name, res in runs:
        gap = res.gap()
        gap_text = "n/a" if gap is None else f"{100 * gap:>9.2e}%"
        p_opt = res.success_probability()
        p_text = "n/a" if p_opt is None else f"{p_opt:>11.5f}"
        print(f"  {name:>21} {res.objective:>11.5f} {gap_text:>10} "
              f"{res.decoded['n_cut']:>10} {res.decoded['cut_weight']:>11.4f} "
              f"{p_text:>11} {res.resources['wall_time_s']:>9.3f}")

    res_sa = runs[1][1]
    decoded = res_sa.decoded
    assignment = np.asarray(decoded["islands"], dtype=int)

    # -- island contents ----------------------------------------------------
    print("\nIsland contents")
    print(f"  {'island':>7} {'buses':>28} {'generation':>11} {'load':>8} {'net [MW]':>10} "
          f"{'connected':>10}")
    for side in (0, 1):
        members = np.flatnonzero(assignment == side)
        bus_numbers = [int(net.bus[i, 0]) for i in members]
        gen_mw = float(net.gen_p_per_bus()[members].sum())
        load_mw = float(net.load_p[members].sum())
        print(f"  {'AB'[side]:>7} {str(bus_numbers):>28} {gen_mw:>11.1f} {load_mw:>8.1f} "
              f"{gen_mw - load_mw:>10.1f} "
              f"{'yes' if decoded['islands_connected'][side] else 'NO':>10}")

    # -- the lines that open, and what they were carrying --------------------
    print("\nLines opened by the split, DC power flow before the split")
    print(f"  {'branch':>10} {'coupling 1/x [pu]':>18} {'flow [MW]':>11} "
          f"{'loading':>9} {'direction':>12}")
    transfer_into_b = 0.0
    for f, t in decoded["cut_lines"]:
        line = branch_of(net, f, t)
        flow = float(dc.flow_mw[line])
        coupling = 1.0 / abs(net.branch[line, 3])
        from_bus, to_bus = int(net.bus[f, 0]), int(net.bus[t, 0])
        # positive flow leaves the island of the from-bus
        signed = flow if assignment[f] == 0 else -flow
        transfer_into_b += signed
        direction = "A to B" if signed > 0 else "B to A"
        print(f"  {from_bus:>4} -{to_bus:>4} {coupling:>18.3f} {flow:>11.2f} "
              f"{dc.loading[line]:>9.3f} {direction:>12}")

    imbalance = decoded["island_power_mw"]
    print(f"\n  net transfer A to B before the split   {transfer_into_b:>8.2f} MW")
    print(f"  island A surplus after the split      {imbalance[0]:>8.2f} MW")
    print(f"  island B surplus after the split      {imbalance[1]:>8.2f} MW")
    system_surplus = float(injection_mw.sum())
    print(
        f"\n  Island B imports {abs(transfer_into_b):.1f} MW before the split, so it must\n"
        f"  add {abs(imbalance[1]):.1f} MW of generation or shed the same load right\n"
        "  after the split to hold its frequency. The scheduled generation of case9\n"
        f"  exceeds its load by {system_surplus:.1f} MW. DC power flow assigns that\n"
        "  difference to the slack bus at bus 1, which sits in island A, so the\n"
        f"  {imbalance[0]:.1f} MW surplus of island A is that {system_surplus:.1f} MW plus the\n"
        f"  {abs(transfer_into_b):.1f} MW it no longer exports."
    )

    # -- figure 1: the split -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    qg.viz.plot_network(
        net,
        islands=assignment,
        cut_edges=decoded["cut_lines"],
        ax=ax,
        title=(f"case9 controlled islanding, {decoded['n_cut']} lines opened, "
               f"surplus {imbalance[0]:+.1f} / {imbalance[1]:+.1f} MW"),
    )
    path_network = FIGDIR / "03_islanding_case9_network.png"
    fig.savefig(path_network, dpi=150, bbox_inches="tight")
    plt.close(fig)

    res_qaoa = runs[2][1]
    print(
        "\n  The QAOA output distribution holds two optimal bitstrings, not one. The\n"
        "  labels A and B carry no physical meaning, so every assignment and its\n"
        "  complement have the same objective. The P(optimum) reported above is the\n"
        "  sum over both."
    )

    # -- figure 2: the QAOA output distribution ------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    qg.viz.plot_distribution(res_qaoa, top=12, ax=ax)
    ax.set_title(f"QAOA p=2 output distribution, P(optimum) = "
                 f"{res_qaoa.success_probability():.3f}")
    path_distribution = FIGDIR / "03_islanding_case9_distribution.png"
    fig.savefig(path_distribution, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nfigures            {path_network.name}, {path_distribution.name}")
    print(f"                   in {FIGDIR}")

    # -- self-validation ------------------------------------------------------
    res_exact = runs[0][1]
    print("\nvalidation")
    print(f"  simulated annealing gap        {res_sa.gap():.3e}")
    print(f"  QAOA p=2 gap                   {res_qaoa.gap():.3e}")
    print(f"  islands connected              {decoded['islands_connected']}")
    print(f"  transfer minus island surplus  "
          f"{abs(transfer_into_b + imbalance[1]):.3e} MW")

    assert abs(res_sa.gap()) < TOL_GAP, f"annealing gap {res_sa.gap()}"
    assert abs(res_qaoa.gap()) < TOL_GAP, f"QAOA gap {res_qaoa.gap()}"
    assert all(decoded["islands_connected"]), "an island is not connected"
    assert res_sa.feasible and res_exact.feasible, "a solver returned a single island"
    assert decoded["n_cut"] == res_exact.decoded["n_cut"], "cut size disagrees with exact"
    assert abs(transfer_into_b + imbalance[1]) < 1e-6, (
        "the pre-split transfer does not match the post-split island surplus"
    )
    assert np.isclose(injection_mw.sum(), sum(imbalance)), (
        "island surpluses do not add up to the system surplus"
    )
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
