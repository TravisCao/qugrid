"""Solver benchmark across three power system QUBOs, three seeds each.

WHAT
    Three decision problems that a grid operator recognizes:
      * unit commitment, 2 thermal units over 2 periods with startup costs;
      * controlled islanding of the WSCC 9-bus system;
      * minimum PMU placement for full observability of the PJM 5-bus system.
    Each is encoded as a QUBO and given to four solvers: exhaustive
    enumeration, simulated annealing, depth-2 QAOA, and uniform random
    sampling. Every combination runs with seeds 0, 1, and 2.

WHY QUANTUM
    A quantum optimizer is only interesting on a problem where it beats the
    classical methods that need no quantum hardware. This benchmark measures
    that comparison directly. The result on these instances is that simulated
    annealing matches the exact optimum everywhere at a fraction of a second,
    and QAOA does not improve on it.

    PMU placement runs on the PJM 5-bus system, not the 9-bus system: the
    9-bus formulation needs 24 binaries (9 placement bits plus 15 slack bits)
    and the built-in statevector simulator holds at most 22 qubits.

EXPECTED OUTPUT
    A mean-and-spread summary, an engineering-units comparison, and a grouped
    bar figure of the mean optimality gap and the mean wall time. Results,
    summary, LaTeX table, and run configuration are written to
    examples/runs/benchmark/. Simulated annealing reaches gap 0 on all three
    problems in under 0.1 s. QAOA reaches gap 0 on islanding and on PMU
    placement, and fails on unit commitment with a mean gap of 255%: the
    constraint penalties stretch that QUBO energy range to 709,000 while the
    cost difference between the best and the second-best schedule is 30, so
    the rescaled cost Hamiltonian cannot separate them. Runtime is about 20
    seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import qugrid as qg  # noqa: E402
from qugrid import bench  # noqa: E402
from qugrid.classical.dispatch import GenParams  # noqa: E402
from qugrid.problems import CombinatorialProblem  # noqa: E402
from qugrid.viz import PALETTE, use_style  # noqa: E402

SEEDS = range(3)
CAPTION = "QuGrid solver benchmark"
FIGDIR = Path(__file__).parent / "figures"
RUNDIR = Path(__file__).parent / "runs" / "benchmark"

SOLVERS = {
    "exact": {},
    "sa": {},
    "qaoa": {"p": 2},
    "random": {"samples": 256},
}

# Engineering quantity reported per problem: decoded key, label, format.
METRICS = {
    "uc-2gen": ("cost", "total cost [$]", "{:.2f}"),
    "island-case9": ("cut_weight", "cut weight [pu]", "{:.3f}"),
    "pmu-case5": ("n_pmu", "PMUs installed", "{:.0f}"),
}


def booktabs(summary: pd.DataFrame, caption: str) -> str:
    """A booktabs table of a summarized benchmark, ready to include in a paper."""
    cols = list(summary.columns)
    align = "ll" + "r" * (len(cols) - 2)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:qugrid-benchmark}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(c.replace("_", " ") for c in cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        cells = [
            row[c] if isinstance(row[c], str) else "--" if pd.isna(row[c]) else f"{row[c]:.4g}"
            for c in cols
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def save_artifacts(df: pd.DataFrame, config: dict) -> tuple[Path, str]:
    """Write results.csv, summary.csv, summary.tex, and config.json.

    ``qugrid.bench.save_run`` writes the same four files and is used when it
    can. Its LaTeX step needs jinja2, because pandas routes
    ``DataFrame.to_latex`` through its Styler from version 2.0 onward, and
    jinja2 is not a QuGrid dependency. The local writer keeps the run
    reproducible in either environment.
    """
    try:
        return bench.save_run(df, RUNDIR, config=config, caption=CAPTION), "qugrid.bench"
    except ImportError:
        RUNDIR.mkdir(parents=True, exist_ok=True)
        summary = bench.summarize(df)
        df.to_csv(RUNDIR / "results.csv", index=False)
        summary.to_csv(RUNDIR / "summary.csv", index=False)
        (RUNDIR / "summary.tex").write_text(booktabs(summary, CAPTION))
        (RUNDIR / "config.json").write_text(json.dumps(config, indent=2))
        return RUNDIR, "local writer (jinja2 not installed)"


def build_problems() -> dict[str, CombinatorialProblem]:
    gens = [
        GenParams("base", pmin=0, pmax=90, c2=0.02, c1=10, c0=50, startup=100),
        GenParams("peaker", pmin=0, pmax=60, c2=0.04, c1=20, c0=30, startup=80),
    ]
    return {
        "uc-2gen": qg.problems.UnitCommitment(gens, [60, 130], power_bits=2),
        "island-case9": qg.problems.Islanding(qg.cases.case9()),
        "pmu-case5": qg.problems.PMUPlacement(qg.cases.case5()),
    }


def main() -> int:
    use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    problems = build_problems()
    print("problem instances")
    for name, prob in problems.items():
        print(f"  {name:<14} {type(prob).__name__:<26} {prob.n:>3} binary variables")
    print(f"\nsolvers: {', '.join(SOLVERS)}   seeds: {list(SEEDS)}\n")

    df = bench.sweep(problems, SOLVERS, seeds=SEEDS, verbose=False)
    summary = bench.summarize(df)

    header = (
        f"{'problem':<14}  {'solver':<7}  {'objective':>12}  {'std':>9}  "
        f"{'gap %':>9}  {'feasible':>9}  {'P(opt) %':>9}  {'time s':>8}"
    )
    print(header)
    print("-" * len(header))
    for _, row in summary.sort_values(["problem", "solver"]).iterrows():
        p_success = row["p_success_mean"]
        p_text = "n/a" if not np.isfinite(p_success) else f"{100 * p_success:.3f}"
        print(
            f"{row['problem']:<14}  {row['solver']:<7}  {row['objective_mean']:>12,.3f}  "
            f"{row['objective_std']:>9,.3f}  {100 * row['gap_mean']:>9.2f}  "
            f"{100 * row['feasible_rate']:>8.0f}%  {p_text:>9}  {row['time_mean_s']:>8.4f}"
        )

    # --------------------------------------------- engineering-units comparison
    print("\nengineering units at seed 0 (exact enumeration is the baseline)")
    header = f"{'problem':<14}  {'metric':<16}"
    header += "".join(f"{name:>12}" for name in SOLVERS)
    print(header)
    print("-" * len(header))
    for name, problem in problems.items():
        key, label, fmt = METRICS[name]
        cells = []
        for solver, opts in SOLVERS.items():
            res = qg.solve(problem, solver=solver, seed=0, **opts)
            value = fmt.format(float(res.decoded[key]))
            cells.append(value if res.feasible else f"{value}*")
        print(f"{name:<14}  {label:<16}" + "".join(f"{c:>12}" for c in cells))
    print("* marks a solution that violates the original constraints.")

    # ----------------------------------------------------------------- artifacts
    config = {
        "solvers": {name: opts for name, opts in SOLVERS.items()},
        "seeds": list(SEEDS),
        "problems": {name: {"n_variables": p.n} for name, p in problems.items()},
    }
    out, writer = save_artifacts(df, config)
    print(f"\nresults    -> {out / 'results.csv'}")
    print(f"summary    -> {out / 'summary.csv'}")
    print(f"LaTeX      -> {out / 'summary.tex'}   [written by {writer}]")
    print(f"config     -> {out / 'config.json'}")

    # ------------------------------------------------------------------- figure
    order = list(problems)
    solver_names = list(SOLVERS)
    gaps = np.array(
        [
            [
                100.0
                * float(
                    summary.loc[
                        (summary["problem"] == prob) & (summary["solver"] == solver),
                        "gap_mean",
                    ].iloc[0]
                )
                for prob in order
            ]
            for solver in solver_names
        ]
    )
    times = np.array(
        [
            [
                float(
                    summary.loc[
                        (summary["problem"] == prob) & (summary["solver"] == solver),
                        "time_mean_s",
                    ].iloc[0]
                )
                for prob in order
            ]
            for solver in solver_names
        ]
    )

    x = np.arange(len(order))
    width = 0.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for i, solver in enumerate(solver_names):
        offset = (i - 1.5) * width
        heights = np.maximum(gaps[i], 0.0)
        ax1.bar(
            x + offset,
            heights,
            width,
            label=solver,
            color=PALETTE[i],
            edgecolor="white",
            linewidth=0.8,
        )
        # A gap of 0 draws no bar, so mark it as a measured result.
        for xi, height in zip(x + offset, heights):
            if height < 1e-9:
                ax1.annotate(
                    "0",
                    (xi, 0.0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=PALETTE[i],
                    fontweight="bold",
                )
        ax2.bar(
            x + offset, times[i], width, label=solver, color=PALETTE[i],
            edgecolor="white", linewidth=0.8,
        )
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_xticks(x, order)
    ax1.set_ylabel("mean optimality gap [%]")
    ax1.set_title("Distance from the exact optimum (0 is best)")
    ax2.set_yscale("log")
    ax2.set_xticks(x, order)
    ax2.set_ylabel("mean wall time [s]")
    ax2.set_title("Cost of the answer")
    fig.suptitle(f"QuGrid solver benchmark, {len(list(SEEDS))} seeds per cell")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGDIR / "07_solver_benchmark_gap.png", dpi=150)
    plt.close(fig)
    print(f"figure     -> {FIGDIR / '07_solver_benchmark_gap.png'}")

    # --------------------------------------------------------------- validation
    exact_gaps = df.loc[df["solver"] == "exact", "gap"].to_numpy()
    assert np.all(np.abs(exact_gaps) < 1e-9), f"exact enumeration has a gap: {exact_gaps}"
    for prob in order:
        sa_gap = summary.loc[
            (summary["problem"] == prob) & (summary["solver"] == "sa"), "gap_mean"
        ].iloc[0]
        rand_gap = summary.loc[
            (summary["problem"] == prob) & (summary["solver"] == "random"), "gap_mean"
        ].iloc[0]
        assert sa_gap <= rand_gap + 1e-9, (
            f"{prob}: simulated annealing ({sa_gap:.4f}) is worse than "
            f"random sampling ({rand_gap:.4f})"
        )
    assert df["gap"].notna().all(), "some runs produced no optimality gap"
    print("\nVALIDATION PASSED: exact enumeration has gap 0 everywhere, and simulated")
    print("annealing is at least as good as random sampling on every problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
