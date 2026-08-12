"""bench.py: sweep -> DataFrame -> summary -> LaTeX -> saved run.

``bench.to_latex`` writes booktabs LaTeX by hand instead of calling pandas'
``DataFrame.to_latex`` (which has required jinja2 unconditionally since
pandas 2.0, and qugrid does not declare jinja2). The tests below run the
full pipeline under the project's own declared dependencies.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import qugrid as qg
from qugrid import bench


@pytest.fixture()
def tiny_df(case9):
    """1 problem x 2 solvers x 2 seeds -- the smallest sweep worth aggregating.

    Uses case9, not toy3: toy3 is a complete 3-node graph, so every
    non-trivial island split must cut 2 of its 3 edges, and the Islanding
    QUBO's (deliberately small, see problems/islanding.py) size-balance term
    is too weak to outweigh that cost -- its true optimum on toy3 is the
    degenerate single-island split, which is_feasible correctly rejects.
    case9's real (non-complete) topology has a feasible global optimum, so
    it is the representative case for a "sweep worked end to end" check.
    """
    problems = {"case9-islanding": qg.problems.Islanding(case9)}
    return bench.sweep(problems, solvers=["exact", "sa"], seeds=range(2), verbose=False)


def test_sweep_shape_and_columns(tiny_df):
    assert len(tiny_df) == 1 * 2 * 2
    expected_cols = {
        "problem", "solver", "seed", "objective", "gap", "feasible",
        "success_probability", "wall_time_s", "n_qubits", "evaluations",
    }
    assert expected_cols.issubset(tiny_df.columns)
    assert set(tiny_df["problem"]) == {"case9-islanding"}
    assert set(tiny_df["solver"]) == {"exact", "sa"}
    assert set(tiny_df["seed"]) == {0, 1}
    assert tiny_df["feasible"].all()


def test_sweep_accepts_per_solver_kwargs_dict(toy3):
    problems = {"toy3-islanding": qg.problems.Islanding(toy3)}
    df = bench.sweep(problems, solvers={"sa": {"n_sweeps": 200}}, seeds=[0], verbose=False)
    assert len(df) == 1
    assert df.loc[0, "solver"] == "sa"


def test_summarize_aggregates_per_problem_and_solver(tiny_df):
    summary = bench.summarize(tiny_df)
    assert set(summary["solver"]) == {"exact", "sa"}
    expected_cols = {
        "problem", "solver", "objective_mean", "objective_std", "gap_mean",
        "feasible_rate", "p_success_mean", "time_mean_s", "runs",
    }
    assert expected_cols.issubset(summary.columns)
    assert (summary["runs"] == 2).all()
    assert (summary["feasible_rate"] == 1.0).all()


def test_to_latex_no_jinja2_needed(tiny_df):
    """The writer is hand-rolled booktabs: no jinja2/Styler import anywhere."""
    tex = bench.to_latex(tiny_df, caption="tiny run", label="tab:tiny")
    assert tex.startswith("\\begin{table}[t]")
    assert tex.rstrip().endswith("\\end{table}")
    assert "\\caption{tiny run}" in tex
    assert "\\label{tab:tiny}" in tex
    assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
    # underscores in problem names must arrive escaped
    assert "case9-islanding" in tex
    assert "objective\\_mean" in tex


def test_to_latex_escapes_and_nan():
    df = pd.DataFrame(
        {
            "problem": ["uc_2x2 & co"],
            "solver": ["sa"],
            "objective_mean": [2908.0],
            "gap_mean": [float("nan")],
        }
    )
    tex = bench.to_latex(df)
    assert "uc\\_2x2 \\& co" in tex
    assert "--" in tex  # NaN cell
    assert "2.91e+03" in tex


def test_save_run_writes_expected_files(tiny_df, tmp_path):
    out = bench.save_run(tiny_df, tmp_path / "run1", config={"note": "unit test"})
    assert out == tmp_path / "run1"
    for fname in ("results.csv", "summary.csv", "summary.tex", "config.json"):
        assert (out / fname).exists()

    saved_results = pd.read_csv(out / "results.csv")
    assert len(saved_results) == len(tiny_df)
    assert set(saved_results["solver"]) == {"exact", "sa"}

    saved_summary = pd.read_csv(out / "summary.csv")
    assert set(saved_summary["solver"]) == {"exact", "sa"}

    tex = (out / "summary.tex").read_text()
    assert tex.startswith("\\begin{table}[t]") and "\\bottomrule" in tex

    meta = json.loads((out / "config.json").read_text())
    assert meta["config"] == {"note": "unit test"}
    assert meta["qugrid_version"] == qg.__version__
    assert "numpy_version" in meta and "python" in meta and "timestamp_utc" in meta
