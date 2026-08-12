"""Optimal PMU placement on the 9-bus and 14-bus systems, solved as a QUBO.

WHAT THIS SOLVES
    A phasor measurement unit (PMU) installed at a bus measures the voltage
    phasor of that bus and the current phasors of every line connected to it,
    which makes the bus and all of its neighbours observable. Full topological
    observability with the fewest units is the minimum dominating set of the
    network graph. This script solves it for the WSCC 9-bus system and the IEEE
    14-bus system. Zero-injection buses are not modeled, so the counts here are
    the plain dominating set numbers.

    The QUBO adds binary slack variables to turn each coverage inequality
    x_i + sum_{j in N(i)} x_j >= 1 into an equality before the penalty
    expansion. The 9-bus system therefore needs 9 placement bits and 15 slack
    bits, and the 14-bus system needs 14 placement bits and 32 slack bits.

WHY QUANTUM IS STUDIED HERE
    PMU placement is a pure binary covering problem with no continuous
    variables, so it reaches annealing hardware with no encoding loss; the
    instances here are small enough to verify exactly, which is what makes them
    useful as a benchmark rather than a demonstration.

EXPECTED OUTPUT
    Three tables print: the two instances and their variable counts, the exact
    minimum against simulated annealing for each, and the buses chosen with
    their observability check. One figure is written:
      examples/figures/04_pmu_placement_networks.png
    The script exits with status 0 when every assertion at the end holds.

    The exact answer comes from PMUPlacement.reference(), which enumerates the
    placement bits only: 2**9 and 2**14 states. Enumerating the full QUBO would
    need 2**24 and 2**46 states, which is why the annealer works on the full
    encoding and the reference does not.
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

# Annealing effort per case. The 14-bus instance has 46 variables and needs
# more sweeps than the default 1500 to reach the exact minimum from seed 0.
SA_SETTINGS = {
    "case9": {"n_sweeps": 2000, "n_restarts": 12},
    "case14": {"n_sweeps": 8000, "n_restarts": 30},
}
EXPECTED_MINIMUM = {"case9": 3, "case14": 4}


def solve_case(name: str) -> dict:
    """Exact minimum dominating set and the annealing answer for one case."""
    net = getattr(qg.cases, name)()
    problem = qg.problems.PMUPlacement(net)
    exact = problem.reference()
    annealed = qg.solve(problem, solver="sa", seed=SEED, **SA_SETTINGS[name])
    n_placement = net.n_bus
    return {
        "name": name,
        "net": net,
        "problem": problem,
        "exact": exact,
        "annealed": annealed,
        "n_placement": n_placement,
        "n_slack": problem.n - n_placement,
    }


def main() -> int:
    qg.viz.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    cases = [solve_case(name) for name in ("case9", "case14")]

    print("Instances")
    print(f"  {'case':>8} {'buses':>6} {'branches':>9} {'placement bits':>15} "
          f"{'slack bits':>11} {'QUBO variables':>15}")
    for case in cases:
        net = case["net"]
        print(f"  {case['name']:>8} {net.n_bus:>6} {net.n_branch:>9} "
              f"{case['n_placement']:>15} {case['n_slack']:>11} {case['problem'].n:>15}")

    print("\nExact minimum against simulated annealing")
    print(f"  {'case':>8} {'exact PMUs':>11} {'annealed PMUs':>14} {'excess':>7} "
          f"{'uncovered buses':>16} {'time [s]':>9}")
    for case in cases:
        exact, annealed = case["exact"], case["annealed"]
        excess = annealed.decoded["n_pmu"] - exact["n_pmu"]
        print(f"  {case['name']:>8} {exact['n_pmu']:>11} "
              f"{annealed.decoded['n_pmu']:>14} {excess:>7} "
              f"{annealed.decoded['n_uncovered']:>16} "
              f"{annealed.resources['wall_time_s']:>9.3f}")

    print("\nChosen buses and observability")
    print(f"  {'case':>8} {'method':>12} {'PMU buses':>26} {'buses observed':>15}")
    for case in cases:
        net = case["net"]
        for method, buses, covered in (
            ("exact", case["exact"]["pmu_buses"], case["exact"]["covered"]),
            ("annealing", case["annealed"].decoded["pmu_buses"],
             case["annealed"].decoded["covered"]),
        ):
            print(f"  {case['name']:>8} {method:>12} {str(buses):>26} "
                  f"{int(np.sum(covered)):>7} of {net.n_bus}")

    print(
        "\n  Each PMU here observes its own bus and every adjacent bus, so the count\n"
        "  is the minimum dominating set of the graph. Real placement studies also\n"
        "  use zero-injection buses and Kirchhoff current law to infer further\n"
        "  phasors, which lowers the count; that rule is not part of this model."
    )

    # -- figure: both networks with the PMU buses marked --------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for ax, case in zip(axes, cases):
        net = case["net"]
        exact = case["exact"]
        highlight = [net.bus_index(b) for b in exact["pmu_buses"]]
        qg.viz.plot_network(
            net,
            highlight_buses=highlight,
            ax=ax,
            title=(f"{case['name']}: {exact['n_pmu']} PMUs observe all "
                   f"{net.n_bus} buses"),
        )
    fig.tight_layout()
    path_networks = FIGDIR / "04_pmu_placement_networks.png"
    fig.savefig(path_networks, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nfigure             {path_networks.name}")
    print(f"                   in {FIGDIR}")

    # -- self-validation -----------------------------------------------------
    print("\nvalidation")
    for case in cases:
        exact, annealed = case["exact"], case["annealed"]
        print(f"  {case['name']:>8} exact {exact['n_pmu']} PMUs, annealed "
              f"{annealed.decoded['n_pmu']} PMUs, uncovered "
              f"{annealed.decoded['n_uncovered']}")

    for case in cases:
        name = case["name"]
        exact, annealed = case["exact"], case["annealed"]
        assert exact["n_uncovered"] == 0, f"{name}: exact placement leaves buses unobserved"
        assert annealed.decoded["n_uncovered"] == 0, (
            f"{name}: annealed placement leaves "
            f"{annealed.decoded['n_uncovered']} buses unobserved"
        )
        assert annealed.feasible, f"{name}: annealed placement is not observable"
        assert exact["n_pmu"] == EXPECTED_MINIMUM[name], (
            f"{name}: exact minimum is {exact['n_pmu']}, expected {EXPECTED_MINIMUM[name]}"
        )
        assert annealed.decoded["n_pmu"] == exact["n_pmu"], (
            f"{name}: annealing used {annealed.decoded['n_pmu']} PMUs against the "
            f"minimum {exact['n_pmu']}"
        )
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
