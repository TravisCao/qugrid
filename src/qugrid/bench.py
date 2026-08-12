"""Paper-ready benchmarking: sweep, aggregate, export.

The loop every quantum-for-power paper runs — problems x solvers x seeds —
in one call, with the outputs reviewers ask for: a tidy DataFrame, a LaTeX
table, and a JSON config that pins every seed and parameter.

>>> import qugrid as qg
>>> from qugrid import bench
>>> problems = {"case9": qg.problems.Islanding(qg.cases.case9())}
>>> df = bench.sweep(problems, solvers=["exact", "sa", "qaoa"], seeds=range(3))
>>> print(bench.summarize(df))
>>> bench.save_run(df, "runs/islanding")  # results.csv + summary.tex + config.json
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qugrid._version import __version__
from qugrid.solvers import solve


def sweep(
    problems: dict[str, object],
    solvers: list[str] | dict[str, dict],
    seeds=range(3),
    verbose: bool = True,
    **common_kwargs,
) -> pd.DataFrame:
    """Run every solver on every problem with every seed.

    ``solvers`` is a list of registry names, or a dict mapping name to extra
    keyword arguments (for example ``{"qaoa": {"p": 3}}``). Deterministic
    solvers still run once per seed; their spread doubles as a sanity check.
    """
    solver_items = (
        [(name, {}) for name in solvers]
        if isinstance(solvers, list)
        else list(solvers.items())
    )
    rows = []
    for label, problem in problems.items():
        for solver_name, extra in solver_items:
            for seed in seeds:
                kwargs = {**common_kwargs, **extra}
                res = solve(problem, solver=solver_name, seed=int(seed), **kwargs)
                gap = res.gap()
                row = {
                    "problem": label,
                    "solver": solver_name,
                    "seed": int(seed),
                    "objective": res.objective,
                    "gap": gap,
                    "feasible": res.feasible,
                    "success_probability": res.success_probability(),
                    "wall_time_s": res.resources.get("wall_time_s"),
                    "n_qubits": res.resources.get("n_qubits"),
                    "evaluations": res.resources.get("evaluations"),
                }
                rows.append(row)
                if verbose:
                    gap_s = f"{100 * gap:.2f}%" if gap is not None else "n/a"
                    print(
                        f"[bench] {label:<12} {solver_name:<7} seed={seed} "
                        f"objective={res.objective:,.4g} gap={gap_s}"
                    )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread per (problem, solver), the table most papers print."""
    agg = df.groupby(["problem", "solver"]).agg(
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        gap_mean=("gap", "mean"),
        feasible_rate=("feasible", "mean"),
        p_success_mean=("success_probability", "mean"),
        time_mean_s=("wall_time_s", "mean"),
        runs=("seed", "count"),
    )
    return agg.reset_index()


def to_latex(df: pd.DataFrame, caption: str = "QuGrid benchmark", label: str = "tab:qugrid"):
    """Booktabs LaTeX for a summarized DataFrame (drop it straight into a paper)."""
    summary = df if "objective_mean" in df.columns else summarize(df)
    body = summary.to_latex(
        index=False, float_format=lambda v: f"{v:,.3g}", escape=True, na_rep="--"
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n" + body + "\\end{table}\n"
    )


def save_run(
    df: pd.DataFrame,
    out_dir: str | Path,
    config: dict | None = None,
    caption: str = "QuGrid benchmark",
) -> Path:
    """Write ``results.csv``, ``summary.csv``, ``summary.tex``, ``config.json``.

    The config records package version, platform, and timestamp so the run
    can be cited and reproduced.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    summarize(df).to_csv(out / "summary.csv", index=False)
    (out / "summary.tex").write_text(to_latex(df, caption=caption))
    meta = {
        "qugrid_version": __version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config or {},
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    return out
