"""Cross-library benchmark: QuGrid's own solvers against dimod, Ocean, Qiskit.

WHAT
    The same three power system QUBOs as script 07 — unit commitment over 2
    units and 2 periods, controlled islanding of the WSCC 9-bus system, and
    minimum PMU placement on the PJM 5-bus system — solved by five solvers
    from three libraries:
      * QuGrid simulated annealing and depth-2 QAOA (pure NumPy, no SDK);
      * dimod.ExactSolver, exhaustive enumeration in the Ocean stack;
      * dwave.samplers.SimulatedAnnealingSampler, Ocean's production annealer;
      * QAOA through qiskit-optimization's MinimumEigenOptimizer.
    Every solver receives the identical QUBO object and returns the identical
    Result, so gap, feasibility, and wall time are measured the same way for
    all of them. Three seeds per cell.

WHY QUANTUM
    Nothing here is a quantum claim. Exhaustive enumeration and simulated
    annealing in the Ocean stack are mature, widely reviewed code; this script
    exists so that no reader has to take QuGrid's internal baselines on faith.
    If the internal exact enumeration and dimod.ExactSolver disagree, the
    encoding is wrong, and every quantum result built on it is worthless.

    Qiskit QAOA runs on islanding only. Unit commitment and PMU placement
    carry constraint penalties that stretch the QUBO energy range far beyond
    the cost differences the algorithm must resolve — script 07 measures that
    failure in detail, and repeating it in a second library adds nothing.

EXPECTED OUTPUT
    A table of mean gap, success probability, and wall time per (problem,
    solver), then the per-library agreement check in engineering units.
    Results, summary, LaTeX table, and run configuration are written to
    examples/runs/cross_library/, and the wall-time-against-gap figure to
    examples/figures/11_cross_library_benchmark.png. Both exact solvers agree
    to 1e-6 on all three problems. Both simulated annealers reach gap 0
    everywhere — Ocean's C++ loop in under 0.02 s per run, the built-in NumPy
    annealer in under 0.1 s. Qiskit QAOA on islanding reaches gap 0 on 2 of
    3 seeds, mean gap 2.2%, at about 0.3 s per run, the same order as the
    built-in statevector QAOA. Runtime is about 25 seconds.

    A missing optional dependency is not an error: the script prints a SKIP
    line naming the extra to install and validates what remains.
"""

from __future__ import annotations

import importlib.util
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
from qugrid.solvers.statevector import MAX_QUBITS  # noqa: E402
from qugrid.viz import BASELINE, MUTED, PALETTE, use_style  # noqa: E402

SEEDS = range(3)
CAPTION = "QuGrid cross-library solver benchmark"
FIGDIR = Path(__file__).parent / "figures"
RUNDIR = Path(__file__).parent / "runs" / "cross_library"

# solver -> (library label, import module, pip extra, solver kwargs)
SOLVERS = {
    "sa": ("qugrid", None, None, {}),
    "qaoa": ("qugrid", None, None, {"p": 2}),
    "dimod-exact": ("dimod", "dimod", "dwave", {}),
    "dwave-sa": ("dwave-samplers", "dwave.samplers", "dwave", {"num_reads": 100}),
    "qiskit-qaoa": ("qiskit-optimization", "qiskit_optimization", "qiskit", {"reps": 2}),
}

# Engineering quantity reported per problem: decoded key, label, format.
METRICS = {
    "uc-2gen": ("cost", "total cost [$]", "{:.2f}"),
    "island-case9": ("cut_weight", "cut weight [pu]", "{:.3f}"),
    "pmu-case5": ("n_pmu", "PMUs installed", "{:.0f}"),
}

MARKERS = {"uc-2gen": "o", "island-case9": "s", "pmu-case5": "^"}


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


def available(solvers: dict) -> tuple[list[str], list[str]]:
    """Split the solver names into those that can run and those that cannot."""
    ready, skipped = [], []
    for name, (_lib, module, extra, _opts) in solvers.items():
        if module is None or importlib.util.find_spec(module) is not None:
            ready.append(name)
        else:
            skipped.append(name)
            print(f"SKIP {name} (pip install qugrid[{extra}])")
    return ready, skipped


def applies(solver: str, problem: CombinatorialProblem, label: str) -> bool:
    """Statevector QAOA is capped by qubit count; Qiskit QAOA runs on islanding."""
    if solver == "qaoa":
        return problem.n <= MAX_QUBITS
    if solver == "qiskit-qaoa":
        return label == "island-case9"
    return True


def main() -> int:
    use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    problems = build_problems()
    ready, skipped = available(SOLVERS)
    if skipped:
        print()

    print("problem instances")
    exact = {}
    for label, prob in problems.items():
        exact[label] = qg.solve(prob, solver="exact")
        print(
            f"  {label:<14} {type(prob).__name__:<16} {prob.n:>3} binary variables   "
            f"exact optimum {exact[label].objective:>12,.4f}"
        )
    print(f"\nsolvers: {', '.join(ready)}   seeds: {list(SEEDS)}\n")

    frames = []
    for label, prob in problems.items():
        plan = {s: SOLVERS[s][3] for s in ready if applies(s, prob, label)}
        frames.append(bench.sweep({label: prob}, plan, seeds=SEEDS, verbose=False))
    df = pd.concat(frames, ignore_index=True)
    df["library"] = df["solver"].map(lambda s: SOLVERS[s][0])
    summary = bench.summarize(df)

    # ------------------------------------------------------------------- table
    header = (
        f"{'problem':<14}  {'library':<19}  {'solver':<12}  {'objective':>12}  "
        f"{'gap %':>8}  {'feasible':>9}  {'P(opt) %':>9}  {'time s':>8}"
    )
    print(header)
    print("-" * len(header))
    for _, row in summary.sort_values(["problem", "solver"]).iterrows():
        p_success = row["p_success_mean"]
        p_text = "n/a" if not np.isfinite(p_success) else f"{100 * p_success:.3f}"
        gap = row["gap_mean"] if abs(row["gap_mean"]) > 1e-9 else 0.0  # hide float noise
        print(
            f"{row['problem']:<14}  {SOLVERS[row['solver']][0]:<19}  {row['solver']:<12}  "
            f"{row['objective_mean']:>12,.4f}  {100 * gap:>8.2f}  "
            f"{100 * row['feasible_rate']:>8.0f}%  {p_text:>9}  {row['time_mean_s']:>8.4f}"
        )

    # ------------------------------------------- agreement in engineering units
    print("\nengineering units at seed 0 (internal exact enumeration is the baseline)")
    columns = ["exact"] + ready
    print(f"{'problem':<14}  {'metric':<16}" + "".join(f"{c:>13}" for c in columns))
    for label, prob in problems.items():
        key, metric, fmt = METRICS[label]
        cells = []
        for solver in columns:
            if solver != "exact" and not applies(solver, prob, label):
                cells.append("--")
                continue
            opts = {} if solver == "exact" else SOLVERS[solver][3]
            res = qg.solve(prob, solver=solver, seed=0, **opts)
            value = fmt.format(float(res.decoded[key]))
            cells.append(value if res.feasible else f"{value}*")
        print(f"{label:<14}  {metric:<16}" + "".join(f"{c:>13}" for c in cells))
    print("-- marks a solver that does not apply; * marks an infeasible solution.")

    # ----------------------------------------------------------------- artifacts
    config = {
        "solvers": {name: {"library": SOLVERS[name][0], **SOLVERS[name][3]} for name in ready},
        "skipped": {name: f"qugrid[{SOLVERS[name][2]}]" for name in skipped},
        "seeds": list(SEEDS),
        "problems": {label: {"n_variables": p.n} for label, p in problems.items()},
    }
    out = bench.save_run(df, RUNDIR, config=config, caption=CAPTION)
    print(f"\nresults    -> {out / 'results.csv'}")
    print(f"summary    -> {out / 'summary.csv'}")
    print(f"LaTeX      -> {out / 'summary.tex'}")
    print(f"config     -> {out / 'config.json'}")

    # ------------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for i, solver in enumerate(ready):
        rows = summary[summary["solver"] == solver]
        for k, (_, row) in enumerate(rows.iterrows()):
            ax.scatter(
                row["time_mean_s"],
                max(100 * row["gap_mean"], 0.0),
                s=150,
                marker=MARKERS[row["problem"]],
                color=PALETTE[i],
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
                label=f"{solver} ({SOLVERS[solver][0]})" if k == 0 else None,
            )
    ax.axhline(0.0, color=BASELINE, lw=1.2, zorder=1)
    ax.annotate(
        "gap 0: the exact optimum",
        (0.99, 0.0),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=MUTED,
    )
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-0.4, 600)  # gaps are non-negative; the room below the axis is waste
    ax.set_xlabel("mean wall time [s]  (log)")
    ax.set_ylabel("mean optimality gap [%]")
    ax.set_title("Same QUBO, five solvers, three libraries")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", fontsize=9, title="color = solver")
    ax.annotate(
        "marker: circle = uc-2gen, square = island-case9, triangle = pmu-case5",
        (0.0, -0.20),
        xycoords="axes fraction",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "11_cross_library_benchmark.png", dpi=150)
    plt.close(fig)
    print(f"figure     -> {FIGDIR / '11_cross_library_benchmark.png'}")

    # --------------------------------------------------------------- validation
    checks = []
    if "dimod-exact" in ready:
        for label in problems:
            objectives = df.loc[
                (df["solver"] == "dimod-exact") & (df["problem"] == label), "objective"
            ].to_numpy()
            deviation = float(np.abs(objectives - exact[label].objective).max())
            assert deviation < 1e-6, (
                f"{label}: dimod.ExactSolver and QuGrid enumeration disagree by {deviation:.3g}"
            )
        checks.append("dimod.ExactSolver matches QuGrid enumeration to 1e-6 on all three problems")
    if "dwave-sa" in ready:
        gap = float(
            summary.loc[
                (summary["solver"] == "dwave-sa") & (summary["problem"] == "island-case9"),
                "gap_mean",
            ].iloc[0]
        )
        assert gap < 1e-9, f"Ocean simulated annealing has gap {100 * gap:.3f}% on islanding"
        checks.append("Ocean simulated annealing reaches the exact optimum on islanding")
    assert df["gap"].notna().all(), "some runs produced no optimality gap"
    classical = df[df["solver"].isin(["dimod-exact", "dwave-sa", "sa"])]
    assert classical["feasible"].all(), "a classical solver violated the original constraints"

    print("\nVALIDATION PASSED")
    for line in checks:
        print(f"  {line}")
    if skipped:
        print(f"  {len(skipped)} solver(s) skipped, listed above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
