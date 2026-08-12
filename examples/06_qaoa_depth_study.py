"""QAOA depth study for controlled islanding of the WSCC 9-bus system.

WHAT
    Controlled islanding splits a stressed network into two self-sufficient
    parts. The objective adds the electrical coupling of the opened lines, the
    squared power imbalance of each island, and a size-balance term. On the
    WSCC 9-bus system this gives a 9-variable QUBO, so exhaustive enumeration
    over all 512 states supplies the exact optimum.

WHY QUANTUM
    QAOA depth p controls how much of the cost Hamiltonian the circuit can
    express. Theory says the variational optimum cannot get worse as p grows,
    because a depth-(p+1) circuit contains every depth-p circuit. This script
    measures what a finite classical optimizer with 3 restarts actually
    delivers, and what the extra depth buys in probability of measuring the
    optimum. No speed advantage is claimed: the classical baseline solves this
    instance exactly in under one millisecond.

EXPECTED OUTPUT
    A table with one row per depth p in (1, 2, 3, 4). The energy expectation
    decreases with p (128.85 -> 94.26 at seed 0). The probability of measuring
    an optimal partition rises from 0.90% to 1.27%, against 0.39% for uniform
    random sampling. Every depth returns the exact optimum after classical
    post-selection over the 16 most probable states. Two figures are written to
    examples/figures/. Runtime is about 3 seconds.
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

DEPTHS = (1, 2, 3, 4)
RESTARTS = 3
SEED = 0
FIGDIR = Path(__file__).parent / "figures"


def main() -> int:
    use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    net = qg.cases.case9()
    problem = qg.problems.Islanding(net)

    # Classical baseline: enumerate all 2^n states of the same QUBO.
    exact = qg.solve(problem, solver="exact")
    energies = problem.qubo.to_ising().all_energies()
    optimum = float(energies.min())
    degeneracy = int(np.sum(np.abs(energies - optimum) <= 1e-9 * max(1.0, abs(optimum))))
    uniform_p = degeneracy / len(energies)

    print(f"network            {net.name}, {net.n_bus} buses, {net.n_branch} branches")
    print(f"QUBO variables     {problem.n} (one per bus)")
    print(f"exact optimum      {optimum:.4f} at {degeneracy} of {len(energies)} states")
    print(f"exact partition    sizes {exact.decoded['sizes']}, "
          f"{exact.decoded['n_cut']} lines cut, "
          f"cut weight {exact.decoded['cut_weight']:.3f} pu")
    print(f"exact island power {exact.decoded['island_power_mw'][0]:+.1f} MW / "
          f"{exact.decoded['island_power_mw'][1]:+.1f} MW\n")

    rows = []
    results = {}
    for p in DEPTHS:
        res = qg.solve(problem, solver="qaoa", p=p, restarts=RESTARTS, seed=SEED)
        results[p] = res
        rows.append(
            {
                "p": p,
                "expectation": float(res.resources["expectation"]),
                "objective": float(res.objective),
                "gap_pct": 100.0 * res.gap(),
                "p_success": float(res.success_probability()),
                "cut_weight": float(res.decoded["cut_weight"]),
                "n_cut": int(res.decoded["n_cut"]),
                "evaluations": int(res.resources["evaluations"]),
                "runtime_s": float(res.resources["wall_time_s"]),
            }
        )

    header = (
        f"{'p':>2}  {'<H>':>10}  {'best obj':>10}  {'gap %':>8}  {'P(opt) %':>9}  "
        f"{'x uniform':>9}  {'cut wt pu':>9}  {'cut':>4}  {'evals':>6}  {'time s':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['p']:>2}  {r['expectation']:>10.4f}  {r['objective']:>10.4f}  "
            f"{r['gap_pct']:>8.2e}  {100 * r['p_success']:>9.3f}  "
            f"{r['p_success'] / uniform_p:>9.2f}  {r['cut_weight']:>9.3f}  "
            f"{r['n_cut']:>4}  {r['evaluations']:>6}  {r['runtime_s']:>7.2f}"
        )
    print(f"\nuniform random sampling gives P(opt) = {100 * uniform_p:.3f}% "
          f"({degeneracy} optimal states of {len(energies)}).")
    print("The reported objective is the lowest energy among the 16 most probable")
    print("states, which is classical post-selection, not a single measurement.")

    # ---------------------------------------------------------------- figures
    expectations = [r["expectation"] for r in rows]
    successes = [100 * r["p_success"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax1.plot(DEPTHS, expectations, marker="o", color=PALETTE[0])
    ax1.axhline(optimum, color=PALETTE[7], lw=1.4, ls=(0, (3, 3)))
    ax1.annotate(
        f"exact optimum {optimum:.1f}",
        (DEPTHS[-1], optimum),
        xytext=(0, 8),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=PALETTE[7],
    )
    ax1.set_xticks(list(DEPTHS))
    ax1.set_xlabel("QAOA depth p")
    ax1.set_ylabel("energy expectation")
    ax1.set_title("Cost expectation falls with depth")

    ax2.plot(DEPTHS, successes, marker="o", color=PALETTE[2])
    ax2.axhline(100 * uniform_p, color=PALETTE[7], lw=1.4, ls=(0, (3, 3)))
    ax2.annotate(
        f"uniform sampling {100 * uniform_p:.2f}%",
        (DEPTHS[-1], 100 * uniform_p),
        xytext=(0, 8),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=PALETTE[7],
    )
    ax2.set_xticks(list(DEPTHS))
    ax2.set_xlabel("QAOA depth p")
    ax2.set_ylabel("P(optimal partition) [%]")
    ax2.set_title("Probability of the optimum rises with depth")
    fig.suptitle(f"QAOA depth study, {net.name} islanding, {RESTARTS} restarts, seed {SEED}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_qaoa_depth_study_depth.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for ax, p in zip(axes, (DEPTHS[0], DEPTHS[-1])):
        qg.viz.plot_distribution(results[p], top=12, ax=ax)
        ax.set_title(f"p = {p}")
    fig.suptitle("Output distribution, 12 most probable states (blue = optimal)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_qaoa_depth_study_distribution.png", dpi=150)
    plt.close(fig)
    print(f"\nfigures -> {FIGDIR}/06_qaoa_depth_study_*.png")

    # ------------------------------------------------------------- validation
    for a, b in zip(expectations[:-1], expectations[1:]):
        assert b <= a + 1e-6, f"expectation increased with depth: {a:.6f} -> {b:.6f}"
    assert successes[-1] > successes[0], "deepest circuit did not improve P(optimum)"
    assert successes[0] > 100 * uniform_p, "QAOA did not beat uniform sampling at p = 1"
    for r in rows:
        assert abs(r["gap_pct"]) < 1e-6, f"p={r['p']} did not recover the exact optimum"
    print("VALIDATION PASSED: expectation is non-increasing in p, P(optimum) grows,")
    print("and every depth recovers the exact optimum.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
